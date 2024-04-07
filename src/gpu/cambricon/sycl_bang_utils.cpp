#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

bool compare_bang_devices(
        const ::sycl::device &lhs, const ::sycl::device &rhs) {
    auto lhs_bang_handle = compat::get_native<CNdev>(lhs);
    auto rhs_bang_handle = compat::get_native<CNdev>(rhs);
    return lhs_bang_handle == rhs_bang_handle;
}

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl