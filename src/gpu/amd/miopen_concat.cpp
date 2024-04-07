#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/ocl/ref_concat.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

namespace {

const impl_list_item_t hip_concat_impl_list[]
        = {impl_list_item_t::concat_type_deduction_helper_t<
                   gpu::ocl::ref_concat_t::pd_t>(),
                nullptr};
} // namespace

const impl_list_item_t *
hip_gpu_engine_impl_list_t::get_concat_implementation_list() {
    return hip_concat_impl_list;
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl