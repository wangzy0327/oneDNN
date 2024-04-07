#ifndef GPU_AMD_MIOPEN_POOLING_HPP
#define GPU_AMD_MIOPEN_POOLING_HPP

#include "common/c_types_map.hpp"
#include "common/pooling_pd.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "gpu/amd/miopen_pooling_impl.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

#include <CL/sycl.hpp>
#include "sycl/sycl_buffer_memory_storage.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_pooling_common_t {
    template <typename pd_t>
    void init_ws(const pd_t *pd, memory_desc_t &ws_md) {
        bool is_fwd = pd->is_fwd();
        memory_desc_wrapper src_wrap(is_fwd ? pd->src_md() : pd->diff_src_md());
        memory_desc_wrapper dst_wrap(is_fwd ? pd->dst_md() : pd->diff_dst_md());

        const auto src_size = src_wrap.nelems();
        const auto dst_size = dst_wrap.nelems();
        const dims_t ws_size = {(dim_t)(src_size + dst_size)};

        dnnl_memory_desc_init_by_tag(
                &ws_md, 1, ws_size, src_wrap.data_type(), format_tag::x);
    }

    status_t init_mem_by_tag(format_tag_t tag, memory_desc_t &md) {
        if (tag == format_tag::undef) { return status::unimplemented; }
        CHECK(memory_desc_init_by_tag(md, tag));
        return status::success;
    }

    format_tag_t get_tag(const memory_desc_t &md) const {
        using namespace format_tag;
        auto tag = memory_desc_matches_one_of_tag(md, ab, abc, abcd,
                abcde, // NCHW derivatives
                ba, bca, bcda, bcdea, cba, cdba,
                cdeba, // IO and spatial derivatives
                acb, acdb, acdeb, // NHWC derivatives
                aBcd16b, aBcde16b, aBcd8b, aBcde8b, aBcd4b,
                aBcde4b); // blocked layouts
        return tag;
    }
};

struct miopen_pooling_fwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public pooling_fwd_pd_t, public miopen_pooling_common_t {
        using pooling_fwd_pd_t::pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_pooling_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace alg_kind;
            using namespace format_tag;

            assert(engine->kind() == engine_kind::gpu);
            auto src_dt = src_md()->data_type;

            bool ok = true && is_fwd();
            ok = ok && set_default_params() == status::success;
            ok = ok
                    && utils::one_of(desc()->prop_kind, forward_training,
                            forward_inference);
            ok = ok
                    && utils::one_of(desc()->alg_kind, pooling_max,
                            pooling_avg_include_padding,
                            pooling_avg_exclude_padding);
            ok = ok && utils::one_of(src_dt, s8, f16, f32);
            ok = ok
                    && IMPLICATION(utils::one_of(src_dt, f16),
                            desc()->prop_kind == forward_inference);
            ok = ok
                    && IMPLICATION(
                            src_dt == s8, desc()->accum_data_type == s32);
            ok = ok && attr()->has_default_values();
            ok = ok && blocking_ok();
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == forward_training;
            if (is_training) init_ws(this, ws_md_);

            if (has_zero_dim_memory()) return status::success;

            pooling_impl_.reset(new miopen_pooling_fwd_impl_t());
            return pooling_impl_->init(this);
        }

        bool blocking_ok() const {
            if (!utils::one_of(src_md()->data_type, data_type::s8)
                    && src_md()->format_desc.blocking.inner_nblks > 0)
                return false;

            if (src_md()->format_desc.blocking.inner_nblks > 1) return false;

            if (utils::one_of(src_md()->data_type, data_type::s8)
                    && src_md()->format_desc.blocking.inner_nblks == 1) {
                return memory_desc_matches_nchw_vect_c(src_md())
                        && memory_desc_matches_nchw_vect_c(dst_md());
            }

            return true;
        }

        std::shared_ptr<miopen_pooling_impl_base_t> pooling_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override{
        memory_desc_wrapper dst_wrap(pd()->dst_md());
        if (dst_wrap.size() == 0) return status::success;

        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        bool is_training = pd()->desc()->prop_kind == prop_kind::forward_training;
        auto wkspace_st = is_training
                ? ctx.output(DNNL_ARG_WORKSPACE)->memory_storage()
                : &memory_storage_t::empty_storage();

        memory_desc_wrapper src_wrap(pd()->src_md());

        auto ws_size = src_wrap.nelems() * src_wrap.data_type_size() + dst_wrap.nelems() * dst_wrap.data_type_size();

        // If src is empty and dst is not, fill dst with 0 (rocm doesn't have func for each type)
        if (src_wrap.size() == 0 && dst_wrap.size() != 0) {
            return hip_stream->interop_task([&](::sycl::handler &cgh) {
                auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);

                compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                    auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                            hip_stream->engine());
                    auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
                    auto dst = sc.memory<void *>(ih, dst_acc);
                    hipMemsetAsync(reinterpret_cast<hipDeviceptr_t>(dst), 0, dst_wrap.size(), hip_stream->get_underlying_stream());
                });
            });
        }

        return hip_stream->interop_task([&](::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);

            std::shared_ptr<
                    ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::write>>
                    wkspace_acc;
            if (!wkspace_st->is_null()) {
                wkspace_acc = std::make_shared<
                        ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::write>>(
                        utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                                wkspace_st)
                                ->buffer()
                                .template get_access<::sycl::access::mode::write>(
                                        cgh));
            }

            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                        hip_stream->engine());
                auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
                auto handle = hip_stream->get_miopen_handle();

                auto x = sc.memory<void *>(ih, src_acc);
                auto y = sc.memory<void *>(ih, dst_acc);
                uint8_t *ws = nullptr;
                if (!wkspace_st->is_null()) {
                    ws = sc.memory<uint8_t *>(ih, *wkspace_acc);
                }

                pd()->pooling_impl_->execute(handle, x, y, ws, ws_size);
            });
        });
    };

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct miopen_pooling_bwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public pooling_bwd_pd_t, public miopen_pooling_common_t {
        using pooling_bwd_pd_t::pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_pooling_bwd_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace alg_kind;
            using namespace format_tag;
            assert(engine->kind() == engine_kind::gpu);

            bool ok = true && !is_fwd()
                    && set_default_params() == status::success
                    && desc()->prop_kind == backward_data
                    && utils::one_of(desc()->alg_kind, pooling_max,
                            pooling_avg_include_padding,
                            pooling_avg_exclude_padding)
                    && (utils::everyone_is(data_type::f32,
                                diff_dst_md()->data_type,
                                diff_src_md()->data_type)
                            || utils::everyone_is(data_type::f16,
                                    diff_dst_md()->data_type,
                                    diff_src_md()->data_type))
                    && attr()->has_default_values() && no_blocking();
            if (!ok) return status::unimplemented;

            init_mem_by_tag(get_tag(diff_dst_md_), diff_src_md_);

            init_ws(this, ws_md_);
            if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;

            if (has_zero_dim_memory()) { return status::success; };

            pooling_impl_.reset(new miopen_pooling_bwd_impl_t());
            return pooling_impl_->init(this);
        }

        bool no_blocking() const {
            return diff_src_md()->format_desc.blocking.inner_nblks
                    + diff_dst_md()->format_desc.blocking.inner_nblks
                    == 0;
        }

        std::shared_ptr<miopen_pooling_impl_base_t> pooling_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override{

        if (has_zero_dims(pd()->diff_src_md()->dims, pd()->diff_src_md()->ndims)
            || has_zero_dims(pd()->diff_dst_md()->dims, pd()->diff_dst_md()->ndims)) {
            return status::success;
        }

        memory_desc_wrapper wrap(pd()->diff_src_md());
        if (wrap.size() == 0) { return status::success; }
        // const auto dst_offset_bytes = wrap.size();

        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([&](::sycl::handler &cgh) {
            auto diff_src_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
            auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
            auto wkspace_acc = CTX_IN_ACCESSOR(DNNL_ARG_WORKSPACE);

            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                        hip_stream->engine());
                auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
                auto handle = hip_stream->get_miopen_handle();

                auto dx = sc.memory<void *>(ih, diff_src_acc);
                auto dy = sc.memory<void *>(ih, diff_dst_acc);
                auto ws = sc.memory<uint8_t *>(ih, wkspace_acc);
                // auto ws_y = ws_x + dst_offset_bytes;

                pd()->pooling_impl_->execute(handle, dx, dy, ws, 0);
            });
        });
    };

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif