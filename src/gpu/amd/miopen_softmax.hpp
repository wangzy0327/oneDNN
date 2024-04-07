#ifndef GPU_AMD_MIOPEN_SOFTMAX_HPP
#define GPU_AMD_MIOPEN_SOFTMAX_HPP

#include <CL/sycl.hpp>

#include "common/primitive.hpp"
#include "common/softmax_pd.hpp"
#include "gpu/amd/miopen_softmax_impl.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"
#include "sycl/sycl_buffer_memory_storage.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_softmax_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public softmax_fwd_pd_t {
        using softmax_fwd_pd_t::softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_softmax_fwd_t);

        status_t init(engine_t *) {
            bool ok = true
                    && utils::one_of(desc()->prop_kind,
                            prop_kind::forward_inference,
                            prop_kind::forward_training)
                    && utils::one_of(desc()->data_desc.data_type,
                            data_type::f32, data_type::f16)
                    // Blocking is supported only for s8 and softmax does not
                    // support it.
                    && src_md()->format_desc.blocking.inner_nblks == 0
                    && dst_md()->format_desc.blocking.inner_nblks == 0
                    && attr()->has_default_values();

            if (!ok) return status::unimplemented;

            softmax_impl_.reset(new miopen_softmax_fwd_impl_t());

            return softmax_impl_->init(this);
        }

        std::shared_ptr<miopen_softmax_impl_base_t> softmax_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override{
        if (memory_desc_wrapper(pd()->desc()->data_desc).has_zero_dim())
            return status::success;

        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([&](::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);

            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                std::vector<void *> args;
                auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                        hip_stream->engine());
                auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
                auto handle = hip_stream->get_miopen_handle();

                args.push_back(sc.memory<void *>(ih, src_acc));
                args.push_back(sc.memory<void *>(ih, dst_acc));

                pd()->softmax_impl_->execute(handle, args.data(), args.size());
            });
        });
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct miopen_softmax_bwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public softmax_bwd_pd_t {
        using softmax_bwd_pd_t::softmax_bwd_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_softmax_bwd_t);

        status_t init(engine_t *) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && utils::one_of(desc()->data_desc.data_type,
                            data_type::f32, data_type::f16)
                    && set_default_formats_common()
                    // Blocking is not supported
                    && dst_md()->format_desc.blocking.inner_nblks == 0
                    && diff_dst_md()->format_desc.blocking.inner_nblks == 0
                    && attr()->has_default_values();

            if (!ok) return status::unimplemented;

            softmax_impl_.reset(new miopen_softmax_bwd_impl_t());

            return softmax_impl_->init(this);
        }

        std::shared_ptr<miopen_softmax_impl_base_t> softmax_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override{
        if (memory_desc_wrapper(pd()->desc()->diff_desc).has_zero_dim())
            return status::success;

        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([&](::sycl::handler &cgh) {
            auto dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DST);
            auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
            auto diff_src_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);

            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                std::vector<void *> args;
                auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                        hip_stream->engine());
                auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
                auto handle = hip_stream->get_miopen_handle();

                args.push_back(sc.memory<void *>(ih, dst_acc));
                args.push_back(sc.memory<void *>(ih, diff_dst_acc));
                args.push_back(sc.memory<void *>(ih, diff_src_acc));

                pd()->softmax_impl_->execute(handle, args.data(), args.size());
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
