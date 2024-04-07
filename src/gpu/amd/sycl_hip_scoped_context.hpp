#ifndef GPU_AMD_SYCL_HIP_SCOPED_CONTEXT_HPP
#define GPU_AMD_SYCL_HIP_SCOPED_CONTEXT_HPP

#include <memory>
#include <thread>

#include <CL/sycl.hpp>
#include <CL/sycl/backend/hip.hpp>    // llvm/sycl暂时还没有通用的pi backend

#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

// Scoped context is required to set the current context of a thread
// to the context of the using queue. The scoped handle class is
// required to put the stream context on top of the hip stack
class hip_sycl_scoped_context_handler_t {
    hipCtx_t original_;
    bool need_to_recover_;

public:
    hip_sycl_scoped_context_handler_t(const sycl_hip_engine_t &);
    // Destruct the scope p_context placed_context_.
    ~hip_sycl_scoped_context_handler_t() noexcept(false);

    // 后面会换用别的api
    template <typename T, typename U>
    inline T memory(const compat::interop_handle &ih, U acc) {
        return compat::get_native_mem<T>(ih, acc);
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif