#ifndef GPU_CAMBRICON_SYCL_BANG_SCOPED_CONTEXT_HPP
#define GPU_CAMBRICON_SYCL_BANG_SCOPED_CONTEXT_HPP

#include <memory>
#include <thread>

#include <CL/sycl.hpp>
#include <CL/sycl/backend/cnrt.hpp>   

#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

// 
class bang_sycl_scoped_context_handler_t {
    CNcontext original_;
    bool need_to_recover_;

public:
    bang_sycl_scoped_context_handler_t(const sycl_bang_engine_t &);
    // Destruct the scope p_context placed_context_.
    ~bang_sycl_scoped_context_handler_t() noexcept(false);

    // 后面会换用别的api
    template <typename T, typename U>
    inline T memory(const compat::interop_handle &ih, U acc) {
        return compat::get_native_mem<T>(ih, acc);
    }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif