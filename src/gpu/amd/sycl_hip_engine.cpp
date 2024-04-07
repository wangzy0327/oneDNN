#include <CL/sycl/backend/hip.hpp>

#include "common/impl_list_item.hpp"
#include "common/utils.hpp"

#include "sycl/sycl_utils.hpp"

#include "gpu/amd/miopen_convolution.hpp"
#include "gpu/amd/miopen_batch_normalization.hpp"
#include "gpu/amd/miopen_matmul.hpp"
#include "gpu/amd/miopen_pooling.hpp"
#include "gpu/amd/miopen_lrn.hpp"
#include "gpu/amd/miopen_binary.hpp"
#include "gpu/amd/miopen_gemm_inner_product.hpp"
#include "gpu/amd/miopen_conv_inner_product.hpp"
#include "gpu/amd/miopen_eltwise.hpp"
#include "gpu/amd/miopen_softmax.hpp"

#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"

#include <stdio.h>

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

bool is_amd_gpu(const cl::sycl::device &dev) {
    constexpr int amd_vendor_id = 0x10DE;
    if(dev.get_info<cl::sycl::info::device::vendor_id>()!= amd_vendor_id)
        printf("vendor id error!\n");
    // return dev.is_gpu();
    return dev.is_gpu()
            && dev.get_info<cl::sycl::info::device::vendor_id>()
            == amd_vendor_id;
}

status_t hip_engine_create(engine_t **engine, engine_kind_t engine_kind,
        const cl::sycl::device &dev, const cl::sycl::context &ctx,
        size_t index) {
    if(engine_kind != dnnl::impl::engine_kind::gpu)
        return status::invalid_arguments;

    std::unique_ptr<amd::sycl_hip_engine_t, engine_deleter_t> hip_engine(
            (new amd::sycl_hip_engine_t(dev, ctx, index)));
    if (!hip_engine) return status::out_of_memory;

    CHECK(hip_engine->init()); 

    *engine = hip_engine.release();
    return status::success;
}

sycl_hip_engine_t::sycl_hip_engine_t(engine_kind_t kind,
        const cl::sycl::device &dev, const cl::sycl::context &ctx, size_t index)
    : base_t(kind, dev, ctx, index) {
    underlying_context_type();
    set_miopen_handle();
    set_rocblas_handle();
}

sycl_hip_engine_t::sycl_hip_engine_t(
        const cl::sycl::device &dev, const cl::sycl::context &ctx, size_t index)
    : sycl_hip_engine_t(engine_kind::gpu, dev, ctx, index) {
    assert(is_amd_gpu(dev));
}

status_t sycl_hip_engine_t::set_rocblas_handle() {
    hip_sycl_scoped_context_handler_t sc(*this);
    rocblas_handle handle;
    CHECK(ROCBLAS_EXECUTE_FUNC_S(rocblas_create_handle, &handle));
    rocblas_handle_.set(
            std::unique_ptr<rocblas_handle, void (*)(rocblas_handle *)>(
                    new rocblas_handle(handle), [](rocblas_handle *h) {
                        if (h != nullptr)
                            ROCBLAS_EXECUTE_FUNC_V(rocblas_destroy_handle, *h);
                        delete h;
                    }));
    handle = nullptr;
    return status::success;
}

status_t sycl_hip_engine_t::set_miopen_handle() {
    hip_sycl_scoped_context_handler_t sc(*this);
    miopenHandle_t handle;
    CHECK(MIOPEN_EXECUTE_FUNC_S(miopenCreate, &handle));
    miopen_handle_.set(
            std::unique_ptr<miopenHandle_t, void (*)(miopenHandle_t *)>(
                    new miopenHandle_t(handle), [](miopenHandle_t *h) {
                        if (h != nullptr)
                            MIOPEN_EXECUTE_FUNC_V(miopenDestroy, *h);
                        delete h;
                    }));
    handle = nullptr;
    return status::success;
}

hipCtx_t sycl_hip_engine_t::get_underlying_context() const {
    return cl::sycl::get_native<cl::sycl::backend::hip>(context());
}

status_t sycl_hip_engine_t::create_stream(stream_t **stream, unsigned flags) {
    return sycl_hip_stream_t::create_stream(stream, this, flags);
}

status_t sycl_hip_engine_t::create_stream(
        stream_t **stream, cl::sycl::queue &queue) {
    return sycl_hip_stream_t::create_stream(stream, this, queue);
}

status_t sycl_hip_engine_t::underlying_context_type() {
    // this is a costly function which take avarage up to 75ms
    // on titanrx. So we must run it once and store the variable
    // in  is_primary_context_;
    hipCtx_t primary;
    hipCtx_t desired
            = cl::sycl::get_native<cl::sycl::backend::hip>(context());
    hipDevice_t hip_device
            = cl::sycl::get_native<cl::sycl::backend::hip>(device());
    CHECK(HIP_EXECUTE_FUNC_S(hipDevicePrimaryCtxRetain, &primary, hip_device));
    CHECK(HIP_EXECUTE_FUNC_S(hipDevicePrimaryCtxRelease, hip_device));
    primary_context_ = (primary == desired);
    return status::success;
}

miopenHandle_t *sycl_hip_engine_t::get_miopen_handle() {
    if (!miopen_handle_.is_set()) set_miopen_handle();
    return miopen_handle_.get().get();
}

rocblas_handle *sycl_hip_engine_t::get_rocblas_handle() {
    if (!rocblas_handle_.is_set()) set_rocblas_handle();
    return rocblas_handle_.get().get();
}

device_id_t sycl_hip_engine_t::device_id() const {
    return device_id_t(static_cast<int>(sycl::backend_t::amd),
            static_cast<uint64_t>(
                    cl::sycl::get_native<cl::sycl::backend::hip>(device())),
            static_cast<uint64_t>(0));
}

void sycl_hip_engine_t::activate_stream_rocblas(stream_t *stream) {
    hip_sycl_scoped_context_handler_t sc(*this);
    auto hip_stream = utils::downcast<sycl_hip_stream_t *>(stream);
    auto streamId = hip_stream->get_underlying_stream();
    assert(context() == hip_stream->queue().get_context());
    hipStream_t current_stream_id = nullptr;
    auto rocblas_handle = get_rocblas_handle();
    ROCBLAS_EXECUTE_FUNC(rocblas_get_stream, *rocblas_handle, &current_stream_id);
    if (current_stream_id != streamId) {
        ROCBLAS_EXECUTE_FUNC(rocblas_set_stream, *rocblas_handle, streamId);
    }
}

void sycl_hip_engine_t::activate_stream_miopen(stream_t *stream) {
    hip_sycl_scoped_context_handler_t sc(*this);
    auto hip_stream = utils::downcast<sycl_hip_stream_t *>(stream);
    auto streamId = hip_stream->get_underlying_stream();
    assert(context() == hip_stream->queue().get_context());
    hipStream_t current_stream_id = nullptr;
    auto miopen_handle = get_miopen_handle();
    MIOPEN_EXECUTE_FUNC(miopenGetStream, *miopen_handle, &current_stream_id);
    if (current_stream_id != streamId) {
        MIOPEN_EXECUTE_FUNC(miopenSetStream, *miopen_handle, streamId);
    }
}

namespace {
using namespace dnnl::impl::data_type;
#define INSTANCE(...) \
    impl_list_item_t( \
            impl_list_item_t::type_deduction_helper_t<__VA_ARGS__::pd_t>())
// clang-format off
const dnnl::impl::impl_list_item_t sycl_hip_impl_list[] = {
        // Elementwise
        INSTANCE(miopen_eltwise_fwd_t),
        INSTANCE(miopen_eltwise_bwd_t),

        // convolution
        INSTANCE(miopen_convolution_fwd_t),
        INSTANCE(miopen_convolution_bwd_data_t),
        INSTANCE(miopen_convolution_bwd_weights_t),

        // Batch Normalization
        INSTANCE(miopen_batch_normalization_fwd_t),
        INSTANCE(miopen_batch_normalization_bwd_t),

        // MatMul
        INSTANCE(miopen_matmul_t),
        
        // Pooling
        INSTANCE(miopen_pooling_fwd_t),
        INSTANCE(miopen_pooling_bwd_t),
        
        // LRN
        INSTANCE(miopen_lrn_fwd_t),
        INSTANCE(miopen_lrn_bwd_t),

        // Binary
        INSTANCE(miopen_binary_t),

        // Inner Product
        INSTANCE(miopen_conv_inner_product_fwd_t),
        INSTANCE(miopen_gemm_inner_product_fwd_t),
        INSTANCE(miopen_conv_inner_product_bwd_data_t),
        INSTANCE(miopen_gemm_inner_product_bwd_data_t),
        INSTANCE(miopen_conv_inner_product_bwd_weights_t),
        INSTANCE(miopen_gemm_inner_product_bwd_weights_t),
        
        // Softmax
        INSTANCE(miopen_softmax_fwd_t),
        INSTANCE(miopen_softmax_bwd_t),
        nullptr,
};
// clang-format on
#undef INSTANCE
} // namespace
const dnnl::impl::impl_list_item_t *sycl_hip_engine_t::get_implementation_list(
        const op_desc_t *) const {
    return sycl_hip_impl_list;
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl