#include "common/engine.hpp"
#include "common/impl_list_item.hpp"
#include "gpu/cambricon/cnnl_concat.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

namespace {

#define INSTANCE(...) \
    impl_list_item_t( \
            impl_list_item_t::concat_type_deduction_helper_t<__VA_ARGS__>())

// clang-format off
const impl_list_item_t bang_concat_impl_list[] = {
        INSTANCE(cnnl_concat_t::pd_t),
        nullptr,
};
// clang-format on

} // namespace

// const impl_list_item_t* bang_gpu_engine_impl_list_t::get_concat_implementation_list(
// const memory_desc_t *, const memory_desc_t *) {
const impl_list_item_t* bang_gpu_engine_impl_list_t::get_concat_implementation_list() {
    return bang_concat_impl_list;
}

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl
