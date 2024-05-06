#ifndef GPU_CAMBRICON_SYCL_BANG_STREAM_HPP
#define GPU_CAMBRICON_SYCL_BANG_STREAM_HPP

#include <cnnl.h>
#include <cn_api.h>

#include "common/engine.hpp"
#include "sycl/sycl_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

// class sycl_bang_stream_t : public dnnl::impl::sycl::sycl_stream_t {
public:
    using base_t = dnnl::impl::sycl::sycl_stream_t;
    // cnnl_handle &get_cnnl_handle();
    cnnlHandle_t &get_cnnl_handle(); 

    static status_t create_stream(stream_t **stream, engine_t *engine, unsigned flags) {
        std::unique_ptr<sycl_bang_stream_t> sycl_stream(new sycl_bang_stream_t(engine, flags));
        if (!sycl_stream) return status::out_of_memory;

        CHECK(sycl_stream->init());
        *stream = sycl_stream.release();
        return status::success;
    }

    static status_t create_stream(
            stream_t **stream, engine_t *engine, ::sycl::queue &queue) {
        unsigned flags;
        CHECK(base_t::init_flags(&flags, queue));

        std::unique_ptr<sycl_bang_stream_t> sycl_stream(
                new sycl_bang_stream_t(engine, flags, queue));

        CHECK(sycl_stream->init());

        *stream = sycl_stream.release();
        return status::success;
    }

    status_t interop_task(std::function<void(::sycl::handler &)>);
    cnrtQueue_t get_underlying_stream();
    CNcontext get_underlying_context();

private:
    status_t init();
    sycl_bang_stream_t(engine_t *engine, unsigned flags, cl::sycl::queue &queue)
        : base_t(engine, flags, queue) {}
    sycl_bang_stream_t(engine_t *engine, unsigned flags)
        : base_t(engine, flags) {}
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
