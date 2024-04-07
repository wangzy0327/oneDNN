#ifndef GPU_CMABRICON_CNNL_CUDA_ELTWISE_HPP
#define GPU_CMABRICON_CNNL_CUDA_ELTWISE_HPP

#include "common/eltwise_pd.hpp"
#include "common/primitive.hpp"
#include "gpu/cambricon/cnnl_eltwise_impl.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_eltwise_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public eltwise_fwd_pd_t {
        using eltwise_fwd_pd_t::eltwise_fwd_pd_t;

        DECLARE_COMMON_PD_T("cnnl:any", cnnl_eltwise_fwd_t);

        status_t init(engine_t *) {
            using namespace alg_kind;

            bool ok = true;
            ok = ok && utils::one_of(desc()->prop_kind,
                            prop_kind::forward_training,
                            prop_kind::forward_inference);
            ok = ok && utils::one_of(desc()->alg_kind, eltwise_relu,
                            eltwise_bounded_relu, eltwise_tanh, eltwise_elu,
                            eltwise_logistic);
            ok = ok && utils::one_of(desc()->data_desc.data_type,
                            data_type::f32, data_type::f16, data_type::s8);
                            
            ok = ok && src_md()->format_desc.blocking.inner_nblks == 0
                    && attr()->has_default_values();

            if (!ok) return status::unimplemented;

            eltwise_fwd_impl_.reset(new cnnl_eltwise_fwd_impl_t());
            return eltwise_fwd_impl_->init(this);
        }
        std::shared_ptr<cnnl_eltwise_impl_base_t> eltwise_fwd_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct cnnl_eltwise_bwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public eltwise_bwd_pd_t {
        using eltwise_bwd_pd_t::eltwise_bwd_pd_t;

        DECLARE_COMMON_PD_T("cnnl:any", cnnl_eltwise_bwd_t);

        status_t init(engine_t *) {
            using namespace alg_kind;
            bool ok = true
                    && desc()->prop_kind == prop_kind::backward_data
                    // Supported algorithms
                    && utils::one_of(desc()->alg_kind, eltwise_bounded_relu,
                            eltwise_relu)
                    // Supported data types
                    && desc()->data_desc.data_type == data_type::f32
                    && IMPLICATION(desc()->alg_kind == eltwise_relu,
                            desc()->alpha == 0)
                    && set_default_formats_common()
                    // Eltwise does not support blocking
                    && src_md()->format_desc.blocking.inner_nblks == 0
                    && diff_dst_md()->format_desc.blocking.inner_nblks == 0
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            eltwise_bwd_impl_.reset(new cnnl_eltwise_bwd_impl_t());
            return eltwise_bwd_impl_->init(this);
        }
        std::shared_ptr<cnnl_eltwise_impl_base_t> eltwise_bwd_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
