#ifndef GPU_AMD_MIOPEN_LRN_HPP
#define GPU_AMD_MIOPEN_LRN_HPP

#include <CL/sycl.hpp>
#include "common/c_types_map.hpp"
#include "common/lrn_pd.hpp"
#include "common/primitive.hpp"
#include "gpu/amd/miopen_lrn_impl.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_lrn_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public lrn_fwd_pd_t {
        using lrn_fwd_pd_t::lrn_fwd_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_lrn_fwd_t);

        status_t init(engine_t *) {
            using namespace data_type;

            bool ok = true && is_fwd()
                    && utils::one_of(desc()->prop_kind,
                            prop_kind::forward_inference,
                            prop_kind::forward_training)
                    && utils::one_of(
                            desc()->alg_kind, alg_kind::lrn_across_channels)
                    && utils::one_of(desc()->data_desc.data_type, f32, f16)
                    && attr()->has_default_values()
                    // Make sure local size is not even (issue #75)
                    && desc_.local_size % 2
                    // lrn does not support blocking
                    && src_md()->format_desc.blocking.inner_nblks == 0;
            if (!ok) return status::unimplemented;

            if (has_zero_dim_memory()) return status::success;

            if (is_training()) { ws_md_ = *dst_md(); }

            lrn_impl_.reset(new miopen_lrn_fwd_impl_t());

            return lrn_impl_->init(this);
        }

        bool is_training() const {
            return desc_.prop_kind == prop_kind::forward_training;
        }

        std::shared_ptr<miopen_lrn_impl_base_t> lrn_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override{
        if (memory_desc_wrapper(pd()->desc()->data_desc).has_zero_dim())
            return status::success;

        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([&](::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto wrksp_acc = pd()->is_training()
                    ? CTX_OUT_ACCESSOR(DNNL_ARG_WORKSPACE)
                    : dst_acc;

            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                        hip_stream->engine());
                auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
                auto handle = hip_stream->get_miopen_handle();
                std::vector<void *> args {sc.memory<void *>(ih, src_acc),
                        sc.memory<void *>(ih, dst_acc),
                        sc.memory<void *>(ih, wrksp_acc)};
                pd()->lrn_impl_->execute(handle, args);
            });
        });
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct miopen_lrn_bwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public lrn_bwd_pd_t {
        using lrn_bwd_pd_t::lrn_bwd_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_lrn_bwd_t);

        status_t init(engine_t *) {
            bool ok = true && !is_fwd()
                    && utils::one_of(
                            desc()->alg_kind, alg_kind::lrn_across_channels)
                    && utils::one_of(desc()->data_desc.data_type,
                            data_type::f16, data_type::f32)
                    && set_default_formats_common()
                    && attr()->has_default_values()
                    && desc_.local_size
                            % 2 // Make sure local size is not even (issue #75)
                    // lrn does not support blocking
                    && src_md()->format_desc.blocking.inner_nblks == 0
                    && diff_dst_md()->format_desc.blocking.inner_nblks == 0;
            if (!ok) return status::unimplemented;
            if (has_zero_dim_memory()) { return status::success; };

            ws_md_ = *diff_dst_md();
            if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;

            lrn_impl_.reset(new miopen_lrn_bwd_impl_t());

            return lrn_impl_->init(this);
        }

        std::shared_ptr<miopen_lrn_impl_base_t> lrn_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override{
        if (memory_desc_wrapper(pd()->desc()->data_desc).has_zero_dim())
            return status::success;

        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([&](::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
            auto diff_src_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
            auto ws_acc = CTX_IN_ACCESSOR(DNNL_ARG_WORKSPACE);

            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                std::vector<void *> args;
                auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                        hip_stream->engine());
                auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
                auto handle = hip_stream->get_miopen_handle();

                args.push_back(sc.memory<void *>(ih, src_acc));
                args.push_back(sc.memory<void *>(ih, ws_acc));
                args.push_back(sc.memory<void *>(ih, diff_src_acc));
                args.push_back(sc.memory<void *>(ih, diff_dst_acc));

                pd()->lrn_impl_->execute(handle, args);
            });
        });
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
