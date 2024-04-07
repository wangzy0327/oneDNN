#ifndef GPU_CAMBRICON_CNNL_BINARY_HPP
#define GPU_CAMBRICON_CNNL_BINARY_HPP

#include "cnnl.h"

#include <CL/sycl.hpp>

#include "common/binary_pd.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/cambricon/cnnl_binary_impl.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_binary_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public binary_pd_t {
        using binary_pd_t::binary_pd_t;

        DECLARE_COMMON_PD_T("cambricon:cnnl:any", cnnl_binary_t);

        status_t init(engine_t *) {
            using namespace data_type;

            bool ok = set_default_params() == status::success;
            ok = ok && check_data_types();
            ok = ok && check_no_blocking();
            ok = ok && check_broadcast();
            ok = ok && attr()->has_default_values(primitive_attr_t::skip_mask_t::scales);   // Not support post operater.
            ok = ok && IMPLICATION(!attr()->scales_.has_default_values(), check_scales_mask());

            if (!ok) return status::unimplemented;

            if (check_for_zero_dims()) return status::success;

            binary_impl_.reset(new cnnl_binary_impl_t());

            return binary_impl_->init(this);
        }

        bool check_for_zero_dims() const {
            return has_zero_dims(src_md(0)->dims, src_md(0)->ndims)
                    || has_zero_dims(src_md(1)->dims, src_md(1)->ndims)
                    || has_zero_dims(dst_md()->dims, dst_md()->ndims);
        }

        bool check_scales_mask() const {
            for (const auto &s : attr()->scales_.scales_) {
                if (s.second.mask_ != 0) return false;
            }
            return true;
        }

        bool check_no_blocking() const {
            // Blocking is not supported by cudnnOpTensor, return false if any
            // blocks are present
            return src_md(0)->format_desc.blocking.inner_nblks
                    + src_md(1)->format_desc.blocking.inner_nblks
                    + dst_md()->format_desc.blocking.inner_nblks
                    == 0;
        }

        bool check_broadcast() const {
            // Source 0 broadcast is not supported
            const int ndims = nstl::min(src_md(0)->ndims, src_md(1)->ndims);
            for (int dim_idx = 0; dim_idx < ndims; dim_idx++) {
                if (src_md(0)->dims[dim_idx] == 1
                        && src_md(0)->dims[dim_idx] != src_md(1)->dims[dim_idx])
                    return false;
            }
            return true;
        }

        bool check_data_types() const {
            using namespace data_type;
            bool inputs_same = src_md(0)->data_type == src_md(1)->data_type;
            dnnl_data_type_t input_type = src_md(0)->data_type;
            dnnl_data_type_t output_type = dst_md()->data_type;

            switch (output_type) {
                case f32:
                    return inputs_same && (input_type == f32);
                case f16:
                    return inputs_same && (input_type == f16);
                case s32:
                    return inputs_same && (input_type == s32);
                default: return false;
            }
            return false;
        }
        std::shared_ptr<cnnl_binary_impl_base_t> binary_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
