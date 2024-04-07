#ifndef GPU_AMD_MIOPEN_CONVOLUTION_PD_HPP
#define GPU_AMD_MIOPEN_CONVOLUTION_PD_HPP

#include "common/convolution_pd.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_convolution_fwd_pd_t : public convolution_fwd_pd_t {
    using convolution_fwd_pd_t::convolution_fwd_pd_t;

    bool set_alg_kind(alg_kind_t kind) { return set_default_alg_kind(kind); }

    bool check_for_zero_dims() const {
        return has_zero_dims(
                       invariant_src_md()->dims, invariant_src_md()->ndims)
                || has_zero_dims(
                        invariant_wei_md(0)->dims, invariant_wei_md(0)->ndims)
                || has_zero_dims(
                        invariant_dst_md()->dims, invariant_dst_md()->ndims);
    }
};
struct miopen_convolution_bwd_data_pd_t : public convolution_bwd_data_pd_t {
    using convolution_bwd_data_pd_t::convolution_bwd_data_pd_t;

    bool set_alg_kind(alg_kind_t kind) { return set_default_alg_kind(kind); }

    bool check_for_zero_dims() const {
        return has_zero_dims(
                       invariant_src_md()->dims, invariant_src_md()->ndims)
                || has_zero_dims(
                        invariant_wei_md(0)->dims, invariant_wei_md(0)->ndims)
                || has_zero_dims(
                        invariant_dst_md()->dims, invariant_dst_md()->ndims);
    }
};
struct miopen_convolution_bwd_weights_pd_t
    : public convolution_bwd_weights_pd_t {
    using convolution_bwd_weights_pd_t::convolution_bwd_weights_pd_t;

    bool set_alg_kind(alg_kind_t kind) { return set_default_alg_kind(kind); }

    bool check_for_zero_dims() const {
        return has_zero_dims(
                       invariant_src_md()->dims, invariant_src_md()->ndims)
                || has_zero_dims(
                        invariant_wei_md(0)->dims, invariant_wei_md(0)->ndims)
                || has_zero_dims(
                        invariant_dst_md()->dims, invariant_dst_md()->ndims);
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
