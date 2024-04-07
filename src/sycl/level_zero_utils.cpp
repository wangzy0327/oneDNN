/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "sycl/level_zero_utils.hpp"
#include "oneapi/dnnl/dnnl_config.h"

#if defined(DNNL_WITH_LEVEL_ZERO)

#include <stdio.h>

#if defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include "windows.h"
#else
#error "Level Zero is supported on Linux and Windows only"
#endif

#include <CL/sycl.hpp>
#include <level_zero/ze_api.h>

#if !defined(__SYCL_COMPILER_VERSION)
#error "Unsupported compiler"
#endif

#if (__SYCL_COMPILER_VERSION < 20200818)
#error "Level Zero is not supported with this compiler version"
#endif

#include "common/c_types_map.hpp"
#include "common/verbose.hpp"

#include "sycl/sycl_utils.hpp"
#if DNNL_USE_SYCL121_API
#include <CL/sycl/backend/level_zero.hpp>
#else
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#endif

#include "sycl/sycl_gpu_engine.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

namespace {

#define ZE_CHECK(f) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            if (get_verbose()) { \
                printf("onednn_verbose,gpu,ze_error,%d\n", (int)(res_)); \
                fflush(0); \
            } \
            return status::runtime_error; \
        } \
    } while (false)

void *find_ze_symbol(const char *symbol) {
#if defined(__linux__)
    void *handle = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
#elif defined(_WIN32)
    // Use LOAD_LIBRARY_SEARCH_SYSTEM32 flag to avoid DLL hijacking issue.
    HMODULE handle = LoadLibraryExA(
            "ze_loader.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#endif
    if (!handle) {
        if (get_verbose())
            printf("onednn_verbose,gpu,error,cannot find Level Zero loader "
                   "library\n");
        assert(!"not expected");
        return nullptr;
    }

#if defined(__linux__)
    void *f = reinterpret_cast<void *>(dlsym(handle, symbol));
#elif defined(_WIN32)
    void *f = reinterpret_cast<void *>(GetProcAddress(handle, symbol));
#endif
    if (!f) {
        if (get_verbose())
            printf("onednn_verbose,gpu,error,cannot find symbol: %s\n", symbol);
        assert(!"not expected");
    }
    return f;
}

template <typename F>
F find_ze_symbol(const char *symbol) {
    return (F)find_ze_symbol(symbol);
}

status_t func_zeModuleCreate(ze_context_handle_t hContext,
        ze_device_handle_t hDevice, const ze_module_desc_t *desc,
        ze_module_handle_t *phModule,
        ze_module_build_log_handle_t *phBuildLog) {
    static auto f = find_ze_symbol<decltype(&zeModuleCreate)>("zeModuleCreate");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hContext, hDevice, desc, phModule, phBuildLog));
    return status::success;
}

status_t func_zeDeviceGetProperties(
        ze_device_handle_t hDevice, ze_device_properties_t *pDeviceProperties) {
    static auto f = find_ze_symbol<decltype(&zeDeviceGetProperties)>(
            "zeDeviceGetProperties");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hDevice, pDeviceProperties));
    return status::success;
}

} // namespace

// This function is called from compatibility layer that ensures compatibility
// with SYCL 2017 API. Once the compatibility layer is removed this function
// can be moved to the anonymous namespace above and a function with SYCL
// data types in its interface can be created to call it.
status_t func_zeKernelCreate(ze_module_handle_t hModule,
        const ze_kernel_desc_t *desc, ze_kernel_handle_t *phKernel) {
    static auto f = find_ze_symbol<decltype(&zeKernelCreate)>("zeKernelCreate");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hModule, desc, phKernel));
    return status::success;
}

// FIXME: Currently SYCL doesn't provide any API to get device UUID so
// we query it directly from Level0 with the zeDeviceGetProperties function.
// The `get_device_uuid` function packs 128 bits of the device UUID, which are
// represented as an uint8_t array of size 16, to 2 uint64_t values.
device_uuid_t get_device_uuid(const ::sycl::device &dev) {
    static_assert(ZE_MAX_DEVICE_UUID_SIZE == 16,
            "ZE_MAX_DEVICE_UUID_SIZE is expected to be 16");

    ze_device_properties_t ze_device_properties;
    auto ze_device = compat::get_native<ze_device_handle_t>(dev);
    auto status = func_zeDeviceGetProperties(ze_device, &ze_device_properties);
    MAYBE_UNUSED(status);
    assert(status == status::success);

    const auto &ze_device_id = ze_device_properties.uuid.id;

    uint64_t uuid[ZE_MAX_DEVICE_UUID_SIZE / sizeof(uint64_t)] = {};
    for (size_t i = 0; i < ZE_MAX_DEVICE_UUID_SIZE; ++i) {
        size_t shift = i % sizeof(uint64_t) * CHAR_BIT;
        uuid[i / sizeof(uint64_t)] |= (((uint64_t)ze_device_id[i]) << shift);
    }
    return device_uuid_t(uuid[0], uuid[1]);
}

status_t sycl_create_kernel_with_level_zero(
        std::unique_ptr<::sycl::kernel> &sycl_kernel,
        const std::string &kernel_name, const sycl_gpu_engine_t *sycl_engine,
        const gpu::compute::binary_t *binary,
        gpu::compute::program_list_t *programs) {
    auto desc = ze_module_desc_t();
    desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    desc.format = ZE_MODULE_FORMAT_NATIVE;
    desc.inputSize = binary->size();
    desc.pInputModule = binary->data();
    desc.pBuildFlags = "";
    desc.pConstants = nullptr;

    ze_module_handle_t ze_module;

    auto ze_device
            = compat::get_native<ze_device_handle_t>(sycl_engine->device());
    auto ze_ctx
            = compat::get_native<ze_context_handle_t>(sycl_engine->context());

    CHECK(func_zeModuleCreate(ze_ctx, ze_device, &desc, &ze_module, nullptr));
    CHECK(compat::make_kernel(sycl_kernel, kernel_name, sycl_engine, ze_module,
            binary, programs));

    return status::success;
}

bool compare_ze_devices(const ::sycl::device &lhs, const ::sycl::device &rhs) {
    auto lhs_ze_handle = compat::get_native<ze_device_handle_t>(lhs);
    auto rhs_ze_handle = compat::get_native<ze_device_handle_t>(rhs);

    return lhs_ze_handle == rhs_ze_handle;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl

#else

namespace dnnl {
namespace impl {
namespace sycl {

device_uuid_t get_device_uuid(const ::sycl::device &) {
    return device_uuid_t(0, 0);
}

status_t sycl_create_kernel_with_level_zero(std::unique_ptr<::sycl::kernel> &,
        const sycl_gpu_engine_t *, const gpu::compute::binary_t *) {
    return status::unimplemented;
}

bool compare_ze_devices(const ::sycl::device &, const ::sycl::device &) {
    return false;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif // DNNL_WITH_LEVEL_ZERO
