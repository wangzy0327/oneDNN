/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef SYCL_SYCL_COMPAT_HPP
#define SYCL_SYCL_COMPAT_HPP

#include "gpu/compute/compute.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_gpu_engine_t;

namespace compat {

status_t make_kernel(std::unique_ptr<::sycl::kernel> &sycl_kernel,
        const std::string &kernel_name, const sycl_gpu_engine_t *sycl_engine,
        void *native_program_handle, const gpu::compute::binary_t *binary,
        gpu::compute::program_list_t *programs);

status_t make_kernel(std::unique_ptr<::sycl::kernel> &sycl_kernel,
        const std::string &kernel_name, const sycl_gpu_engine_t *sycl_engine,
        const gpu::compute::binary_t *binary,
        const gpu::compute::program_list_t *programs);

std::function<void(void *)> get_program_list_deleter();

void *get_native(const ::sycl::device &dev);
void *get_native(const ::sycl::context &ctx);

template <typename native_object_t, typename sycl_object_t>
native_object_t get_native(const sycl_object_t &sycl_object) {
    return reinterpret_cast<native_object_t>(get_native(sycl_object));
}

// Automatically use host_task if it is supported by compiler,
// otherwise fall back to codeplay_host_task.
template <typename H, typename F>
inline auto host_task_impl(H &cgh, F &&f, int) -> decltype(cgh.host_task(f)) {
    cgh.host_task(f);
}

template <typename H, typename F>
inline auto host_task_impl(H &cgh, F &&f, long)
        -> decltype(cgh.codeplay_host_task(f)) {
    cgh.codeplay_host_task(f);
}

template <typename H, typename F>
inline void host_task(H &cgh, F &&f) {
    // Third argument is 0 (int) which prefers the
    // host_task option if both are available.
    host_task_impl(cgh, f, 0);
}

bool is_fp64_supported(const ::sycl::device &dev);
uint64_t init_extensions(const ::sycl::device &dev);

#if DNNL_USE_SYCL121_API
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
constexpr auto target_device = ::sycl::target::global_buffer;
#pragma clang diagnostic pop
#else
constexpr auto target_device = ::sycl::target::device;
#endif

} // namespace compat
} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
