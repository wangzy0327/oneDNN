#include "common/engine.hpp"
#include "common/impl_list_item.hpp"
#include "gpu/cambricon/cnnl_reorder.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

namespace {

#define INSTANCE(...) \
    impl_list_item_t( \
            impl_list_item_t::reorder_type_deduction_helper_t<__VA_ARGS__>())

// clang-format off
const impl_list_item_t bang_reorder_impl_list[] = {
        INSTANCE(cnnl_reorder_t::pd_t),
        nullptr,
};
// clang-format on

} // namespace

const impl_list_item_t *
bang_gpu_engine_impl_list_t::get_reorder_implementation_list(
        const memory_desc_t *, const memory_desc_t *) {
    return bang_reorder_impl_list;
}

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl
