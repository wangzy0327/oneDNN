#include "gpu/amd/sycl_hip_stream.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

rocblas_handle &sycl_hip_stream_t::get_rocblas_handle() {
    auto e = utils::downcast<sycl_hip_engine_t *>(engine());
    e->activate_stream_rocblas(this);
    return *(e->get_rocblas_handle());
}

miopenHandle_t &sycl_hip_stream_t::get_miopen_handle() {
    auto e = utils::downcast<sycl_hip_engine_t *>(engine());
    e->activate_stream_miopen(this);
    return *(e->get_miopen_handle());
}

hipStream_t sycl_hip_stream_t::get_underlying_stream() {
    return compat::get_native<hipStream_t>(*queue_);
}

hipCtx_t sycl_hip_stream_t::get_underlying_context() {
    return compat::get_native<hipCtx_t>(queue_->get_context());
}

status_t sycl_hip_stream_t::init() {
    if ((flags() & stream_flags::in_order) == 0
            && (flags() & stream_flags::out_of_order) == 0)
        return status::invalid_arguments;

    // If queue_ is not set then construct it
    auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine());
    auto status = status::success;

    if (!queue_) {
        auto &sycl_ctx = sycl_engine.context();
        auto &sycl_dev = sycl_engine.device();
        if (!sycl_engine.is_service_stream_created())
            queue_.reset(new ::sycl::queue(sycl_ctx, sycl_dev));
        else {
            stream_t *service_stream;
            CHECK(sycl_engine.get_service_stream(service_stream));
            auto sycl_stream = utils::downcast<sycl_stream_t *>(service_stream);
            queue_.reset(new ::sycl::queue(sycl_stream->queue()));
        }
    } else {
        auto queue_streamId = get_underlying_stream();
        auto sycl_dev = queue().get_device();
        bool args_ok
                = engine()->kind() == engine_kind::gpu && sycl_dev.is_gpu();
        if (!args_ok) return status::invalid_arguments;

        auto queue_context = get_underlying_context();
        hipDevice_t queue_device = compat::get_native<hipDevice_t>(sycl_dev);

        auto engine_context = sycl_engine.get_underlying_context();
        auto engine_device = compat::get_native<hipDevice_t>(sycl_engine.device());

        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));
        auto hip_stream
                = utils::downcast<sycl_hip_stream_t *>(service_stream);
        auto engine_streamId = hip_stream->get_underlying_stream();
        status = ((engine_device != queue_device)
                         || (engine_context != queue_context)
                         || (engine_streamId != queue_streamId))
                ? status::invalid_arguments
                : status::success;
    }

    return status;
}

status_t sycl_hip_stream_t::interop_task(
        std::function<void(::sycl::handler &)> sycl_hip_interop_) {
    try {
        this->set_deps({queue().submit(
                [&](::sycl::handler &cgh) { sycl_hip_interop_(cgh); })});
        return status::success;
    } catch (std::runtime_error &e) {
        error::wrap_c_api(status::runtime_error, e.what());
        return status::runtime_error;
    }
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl