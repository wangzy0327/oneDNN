/*******************************************************************************
* 这个源文件定义了一个继承自stream_t（也即dnnl_stream）的子类，
* 继承关系为sycl_amd_stream_t::sycl_stream_t::compute_stream_t::stream_t
* 根类stream_t中包含了该stream所归属的engine_t
* sycl_hip_stream_t是该stream根类的叶类，由特定的engine叶类创建
* sycl_hip_stream_t的核心结构是其父类中的sycl队列Queue，队列中包含了一系列有序的任务
* sycl_hip_stream_t要服务的对象是具体的primitive叶类，要实现的特定api包括：
*   1、从归属的engine中获取amd库的handle；
*   2、提供向自身sycl队列中提交任务的接口；
*******************************************************************************/
#ifndef GPU_AMD_SYCL_HIP_STREAM_HPP
#define GPU_AMD_SYCL_HIP_STREAM_HPP

#include <rocblas.h>
#include <miopen/miopen.h>
#include <hip/hip_runtime.h>

#include "common/engine.hpp"
#include "sycl/sycl_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

class sycl_hip_stream_t : public dnnl::impl::sycl::sycl_stream_t {
public:
    using base_t = dnnl::impl::sycl::sycl_stream_t;
    rocblas_handle &get_rocblas_handle();
    miopenHandle_t &get_miopen_handle(); 

    static status_t create_stream(
            stream_t **stream, engine_t *engine, unsigned flags) {
        std::unique_ptr<sycl_hip_stream_t> sycl_stream(
                new sycl_hip_stream_t(engine, flags));
        if (!sycl_stream) return status::out_of_memory;

        CHECK(sycl_stream->init());
        *stream = sycl_stream.release();
        return status::success;
    }

    static status_t create_stream(
            stream_t **stream, engine_t *engine, ::sycl::queue &queue) {
        unsigned flags;
        CHECK(base_t::init_flags(&flags, queue));

        std::unique_ptr<sycl_hip_stream_t> sycl_stream(
                new sycl_hip_stream_t(engine, flags, queue));

        CHECK(sycl_stream->init());

        *stream = sycl_stream.release();
        return status::success;
    }

    status_t interop_task(std::function<void(::sycl::handler &)>);
    hipStream_t get_underlying_stream();
    hipCtx_t get_underlying_context();

private:
    status_t init();
    sycl_hip_stream_t(engine_t *engine, unsigned flags, cl::sycl::queue &queue)
        : base_t(engine, flags, queue) {}
    sycl_hip_stream_t(engine_t *engine, unsigned flags)
        : base_t(engine, flags) {}
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
