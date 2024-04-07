#ifndef GPU_CAMBRICON_SYCL_BANG_COMPAT_HPP
#define GPU_CAMBRICON_SYCL_BANG_COMPAT_HPP


#include <CL/sycl/backend/cnrt.hpp>

#include "sycl/sycl_compat.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {
namespace compat {

#if DNNL_USE_SYCL121_API
using interop_handle = ::sycl::interop_handler;
template <typename T, typename U>
T get_native_mem(const interop_handle &ih, U acc) {
    return reinterpret_cast<T>(ih.get_mem<::sycl::backend::cnrt>(acc));
}

template <typename T>
void host_task(::sycl::handler &cgh, const T &task) {
    cgh.interop_task(task);
}

template <typename native_object_t, typename sycl_object_t>
native_object_t get_native(const sycl_object_t &sycl_object) {
    auto handle = sycl_object.template get_native<::sycl::backend::cnrt>();
    return reinterpret_cast<native_object_t>(handle);
}

#else
using interop_handle = ::sycl::interop_handle;
template <typename T, typename U>
T get_native_mem(const interop_handle &ih, U acc) {
    return reinterpret_cast<T>(ih.get_native_mem<::sycl::backend::cnrt>(acc));
}

template <typename T>
void host_task(::sycl::handler &cgh, const T &task) {
    cgh.host_task(task);
}

template <typename native_object_t, typename sycl_object_t>
native_object_t get_native(const sycl_object_t &sycl_object) {
    auto handle = ::sycl::get_native<::sycl::backend::cnrt>(sycl_object);
    return reinterpret_cast<native_object_t>(handle);
}
#endif

} // namespace compat
} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif