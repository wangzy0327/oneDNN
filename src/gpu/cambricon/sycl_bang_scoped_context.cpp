#include "gpu/cambricon/sycl_bang_scoped_context.hpp"


namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

bang_sycl_scoped_context_handler_t::bang_sycl_scoped_context_handler_t(
        const sycl_bang_engine_t &engine)
    : need_to_recover_(false) {
    try {
        auto desired = engine.get_underlying_context();
        BANG_EXECUTE_FUNC(cnCtxGetCurrent, &original_);

        if (original_ != desired) {
            // Sets the desired context as the active one for the thread
            BANG_EXECUTE_FUNC(cnCtxSetCurrent, desired);
            need_to_recover_
                    = !(original_ == nullptr && engine.has_primary_context());
        }
    } catch (const std::runtime_error &e) {
        error::wrap_c_api(status::runtime_error, e.what());
    }
}

bang_sycl_scoped_context_handler_t::
        ~bang_sycl_scoped_context_handler_t() noexcept(false) {
    try {
        if (need_to_recover_) { BANG_EXECUTE_FUNC(cnCtxSetCurrent, original_); }
    } catch (const std::runtime_error &e) {
        error::wrap_c_api(status::runtime_error, e.what());
    }
}

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl