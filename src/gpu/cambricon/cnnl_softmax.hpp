#ifndef GPU_CAMBRICON_CNNL_SOFTMAX_HPP
#define GPU_CAMBRICON_CNNL_SOFTMAX_HPP

#include "cnnl.h"

#include <CL/sycl.hpp>

#include "common/primitive.hpp"
#include "common/softmax_pd.hpp"
#include "gpu/cambricon/cnnl_softmax_impl.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_softmax_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public softmax_fwd_pd_t {
        using softmax_fwd_pd_t::softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T("bang:cnnl:any", cnnl_softmax_fwd_t);

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

            softmax_impl_.reset(new cnnl_softmax_fwd_impl_t());

            return softmax_impl_->init(this);
        }

        std::shared_ptr<cnnl_softmax_impl_base_t> softmax_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct cnnl_softmax_bwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public softmax_bwd_pd_t {
        using softmax_bwd_pd_t::softmax_bwd_pd_t;

        DECLARE_COMMON_PD_T("bang:cnnl:any", cnnl_softmax_bwd_t);

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

            softmax_impl_.reset(new cnnl_softmax_bwd_impl_t());

            return softmax_impl_->init(this);
        }

        std::shared_ptr<cnnl_softmax_impl_base_t> softmax_impl_;
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
