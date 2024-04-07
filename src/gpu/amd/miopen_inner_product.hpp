#ifndef GPU_AMD_MIOPEN_INNER_PRODUCT_HPP
#define GPU_AMD_MIOPEN_INNER_PRODUCT_HPP

#include <CL/sycl.hpp>

#include "common/c_types_map.hpp"
#include "common/inner_product_pd.hpp"
#include "common/primitive.hpp"
#include "gpu/amd/miopen_inner_product_impl.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_inner_product_fwd_t : public primitive_t {
public:
    using primitive_t::primitive_t;

    struct pd_t : public inner_product_fwd_pd_t {
        using inner_product_fwd_pd_t::inner_product_fwd_pd_t;
        std::shared_ptr<miopen_inner_product_impl_base_t> inner_product_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override{
        if (pd()->has_zero_dim_memory()) return status::success;

        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([&](::sycl::handler &cgh) {
            using scratch_acc_t = ::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::read_write, sycl::compat::target_device>;
            using read_acc_t = ::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::read, sycl::compat::target_device>;
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto wei_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            std::shared_ptr<read_acc_t> bias_acc;
            if (pd()->with_bias()) {
                bias_acc = std::make_shared<read_acc_t>(
                        CTX_IN_ACCESSOR(DNNL_ARG_BIAS));
            }
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            std::shared_ptr<scratch_acc_t> ip_scratch_acc;
            std::shared_ptr<scratch_acc_t> spacial_scratch_acc;
            std::shared_ptr<scratch_acc_t> scaled_bias_scratch_acc;
            if (pd()->inner_product_impl_->ip_using_scratchpad()) {
                ip_scratch_acc = std::make_shared<
                        scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                        memory_tracking::names::key_iprod_int_dat_in_acc_dt));
            }
            if (pd()->inner_product_impl_->need_to_transform_filter()) {
                spacial_scratch_acc = std::make_shared<scratch_acc_t>(
                        CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
            }
            if (pd()->inner_product_impl_->conv_using_scale_scratchpad()) {
                scaled_bias_scratch_acc
                        = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                                memory_tracking::names::key_conv_adjusted_scales));
            }
            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                        hip_stream->engine());
                auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
                auto miopen_handle = hip_stream->get_miopen_handle();
                auto rocblas_handle = hip_stream->get_rocblas_handle();

                std::vector<void *> args;

                args.push_back(sc.memory<void *>(ih, src_acc));
                args.push_back(sc.memory<void *>(ih, wei_acc));
                args.push_back(
                        ((pd()->with_bias()) ? sc.memory<void *>(ih, *bias_acc)
                                            : nullptr));
                args.push_back(sc.memory<void *>(ih, dst_acc));
                args.push_back((pd()->inner_product_impl_->ip_using_scratchpad()
                                ? sc.memory<void *>(ih, *ip_scratch_acc)
                                : nullptr));
                args.push_back((
                        pd()->inner_product_impl_->need_to_transform_filter()
                                ? sc.memory<void *>(ih, *spacial_scratch_acc)
                                : nullptr));
                args.push_back((
                        pd()->inner_product_impl_->conv_using_scale_scratchpad()
                                ? sc.memory<void *>(ih, *scaled_bias_scratch_acc)
                                : nullptr));
                pd()->inner_product_impl_->execute(
                        miopen_handle, rocblas_handle, args);
            });
        });        
    }
    virtual const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }
};

struct miopen_inner_product_bwd_data_t : public primitive_t {
public:
    using primitive_t::primitive_t;

    struct pd_t : public inner_product_bwd_data_pd_t {
        using inner_product_bwd_data_pd_t::inner_product_bwd_data_pd_t;

        std::shared_ptr<miopen_inner_product_impl_base_t> inner_product_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override{
        if (pd()->has_zero_dim_memory()) return status::success;
        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([&](::sycl::handler &cgh) {
            using scratch_acc_t = ::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::read_write, sycl::compat::target_device>;
            auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
            auto wei_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto diff_src_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
            std::shared_ptr<scratch_acc_t> ip_scratch_acc;
            std::shared_ptr<scratch_acc_t> spacial_scratch_acc;
            if (pd()->inner_product_impl_->ip_using_scratchpad()) {
                ip_scratch_acc = std::make_shared<
                        scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                        memory_tracking::names::key_iprod_int_dat_in_acc_dt));
            }
            if (pd()->inner_product_impl_->need_to_transform_filter()) {
                spacial_scratch_acc = std::make_shared<scratch_acc_t>(
                        CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
            }
            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                        hip_stream->engine());
                auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
                auto miopen_handle = hip_stream->get_miopen_handle();
                auto rocblas_handle = hip_stream->get_rocblas_handle();

                std::vector<void *> args;

                args.push_back(sc.memory<void *>(ih, diff_src_acc));
                args.push_back(sc.memory<void *>(ih, wei_acc));
                args.push_back(sc.memory<void *>(ih, diff_dst_acc));
                args.push_back((pd()->inner_product_impl_->ip_using_scratchpad()
                                ? sc.memory<void *>(ih, *ip_scratch_acc)
                                : nullptr));
                args.push_back((
                        pd()->inner_product_impl_->need_to_transform_filter()
                                ? sc.memory<void *>(ih, *spacial_scratch_acc)
                                : nullptr));
                pd()->inner_product_impl_->execute(
                        miopen_handle, rocblas_handle, args);
            });
        });        
    }
    virtual const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }
};

struct miopen_inner_product_bwd_weights_t : public primitive_t {
public:
    using primitive_t::primitive_t;
    struct pd_t : public inner_product_bwd_weights_pd_t {
        using inner_product_bwd_weights_pd_t::inner_product_bwd_weights_pd_t;

        std::shared_ptr<miopen_inner_product_impl_base_t> inner_product_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override{
        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        if (pd()->has_zero_dim_memory()) {
            auto wei_sz = memory_desc_wrapper(pd()->diff_weights_md(0)).size();
            size_t bias_sz = (pd()->with_bias()
                            ? memory_desc_wrapper(pd()->diff_weights_md(1)).size()
                            : 0);

            if (wei_sz != 0) {
                auto status = hip_stream->interop_task([&](::sycl::handler &cgh) {
                    auto diff_wei_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_WEIGHTS);
                    cgh.fill(diff_wei_acc, static_cast<uint8_t>(0));
                });
                if (status != status::success) return status;
            }
            if (bias_sz != 0) {
                auto status = hip_stream->interop_task([&](::sycl::handler &cgh) {
                    auto diff_bia_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_BIAS);
                    cgh.fill(diff_bia_acc, static_cast<uint8_t>(0));
                });
                if (status != status::success) return status;
            }
            return status::success;
        }

        return hip_stream->interop_task([&](::sycl::handler &cgh) {
            using scratch_acc_t = ::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::read_write, sycl::compat::target_device>;
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
            auto diff_wei_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_WEIGHTS);
            using write_acc_t = ::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::write, sycl::compat::target_device>;
            std::shared_ptr<write_acc_t> diff_bias_acc;
            if (pd()->with_bias()) {
                diff_bias_acc = std::make_shared<write_acc_t>(
                        CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_BIAS));
            }
            std::shared_ptr<scratch_acc_t> ip_scratch_acc;
            std::shared_ptr<scratch_acc_t> spacial_scratch_acc;
            if (pd()->inner_product_impl_->ip_using_scratchpad()) {
                ip_scratch_acc = std::make_shared<
                        scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                        memory_tracking::names::key_iprod_int_dat_in_acc_dt));
            }
            if (pd()->inner_product_impl_->need_to_transform_filter()) {
                spacial_scratch_acc = std::make_shared<scratch_acc_t>(
                        CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
            }
            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                        hip_stream->engine());
                auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
                auto miopen_handle = hip_stream->get_miopen_handle();
                auto rocblas_handle = hip_stream->get_rocblas_handle();
                std::vector<void *> args;

                args.push_back(sc.memory<void *>(ih, src_acc));
                args.push_back(sc.memory<void *>(ih, diff_dst_acc));
                args.push_back(sc.memory<void *>(ih, diff_wei_acc));
                args.push_back(
                        ((pd()->with_bias()) ? sc.memory<void *>(ih, *diff_bias_acc)
                                            : nullptr));

                args.push_back((pd()->inner_product_impl_->ip_using_scratchpad()
                                ? sc.memory<void *>(ih, *ip_scratch_acc)
                                : nullptr));
                args.push_back((
                        pd()->inner_product_impl_->need_to_transform_filter()
                                ? sc.memory<void *>(ih, *spacial_scratch_acc)
                                : nullptr));
                pd()->inner_product_impl_->execute(miopen_handle, rocblas_handle, args);
            });
        });        
    }

    virtual const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
