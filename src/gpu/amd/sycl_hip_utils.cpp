#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

bool compare_hip_devices(
        const ::sycl::device &lhs, const ::sycl::device &rhs) {
    auto lhs_hip_handle = compat::get_native<hipDevice_t>(lhs);
    auto rhs_hip_handle = compat::get_native<hipDevice_t>(rhs);
    return lhs_hip_handle == rhs_hip_handle;
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl