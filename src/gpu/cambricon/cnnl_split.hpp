#ifndef GPU_CAMBRICON_CNNL_SPLIT_HPP
#define GPU_CAMBRICON_CNNL_SPLIT_HPP

#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/split_pd.hpp"
#include "gpu/cambricon/cnnl_split_impl.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_split_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public split_pd_t {
        using split_pd_t::split_pd_t;
        DECLARE_COMMON_PD_T("bang:cnnl:any", cnnl_split_t);

        status_t init(engine_t *engine) {
            bool ok = (engine->kind() == engine_kind::gpu);
            if (!ok) return status::unimplemented;

            split_.reset(new cnnl_split_impl_t());
            return split_->init(engine, this);
        }
        
        std::shared_ptr<cnnl_split_impl_t> split_;
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
