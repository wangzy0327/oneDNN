#include <CL/sycl/backend/cnrt.hpp>

#include "common/impl_list_item.hpp"
#include "common/utils.hpp"

#include "sycl/sycl_utils.hpp"

#include "gpu/cambricon/cnnl_convolution.hpp"
#include "gpu/cambricon/cnnl_matmul.hpp"
#include "gpu/cambricon/cnnl_split.hpp"
#include "gpu/cambricon/cnnl_lrn.hpp"
//#include "gpu/cambricon/cnnl_inner_product_gemm.hpp"
#include "gpu/cambricon/cnnl_batch_normalization.hpp"
#include "gpu/cambricon/cnnl_eltwise.hpp"
#include "gpu/cambricon/cnnl_softmax.hpp"
#include "gpu/cambricon/cnnl_binary.hpp"
#include "gpu/cambricon/cnnl_pooling.hpp"

#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_scoped_context.hpp"
#include "gpu/cambricon/sycl_bang_stream.hpp"

#include <stdio.h>

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {


bool is_cambricon_gpu(const cl::sycl::device &dev) {
    constexpr int cambricon_vendor_id = 0xcabc;
    if(dev.get_info<cl::sycl::info::device::vendor_id>()!= cambricon_vendor_id)
        printf("vendor id error!\n");
    // return dev.is_gpu();
    return dev.is_gpu()
            && dev.get_info<cl::sycl::info::device::vendor_id>()
            == cambricon_vendor_id;
}

status_t bang_engine_create(engine_t **engine, engine_kind_t engine_kind,
        const cl::sycl::device &dev, const cl::sycl::context &ctx,
        size_t index) {
    if(engine_kind != dnnl::impl::engine_kind::gpu)
        return status::invalid_arguments;

    std::unique_ptr<cambricon::sycl_bang_engine_t, engine_deleter_t> bang_engine(
            (new cambricon::sycl_bang_engine_t(dev, ctx, index)));
    if (!bang_engine) return status::out_of_memory;

    CHECK(bang_engine->init()); 

    *engine = bang_engine.release();
    return status::success;
}

sycl_bang_engine_t::sycl_bang_engine_t(engine_kind_t kind,
        const cl::sycl::device &dev, const cl::sycl::context &ctx, size_t index)
    : base_t(kind, dev, ctx, index) {
    underlying_context_type();
    set_cnnl_handle();
    //set_cnnl_handle();
}

sycl_bang_engine_t::sycl_bang_engine_t(
        const cl::sycl::device &dev, const cl::sycl::context &ctx, size_t index)
    : sycl_bang_engine_t(engine_kind::gpu, dev, ctx, index) {
    assert(is_cambricon_gpu(dev));
}

status_t sycl_bang_engine_t::set_cnnl_handle() {
    bang_sycl_scoped_context_handler_t sc(*this);
    cnnlHandle_t handle;
    cnnlStatus_t err = cnnlCreate(&handle);
    if (err != CNNL_STATUS_SUCCESS) { return cnnl_to_dnnl_status(err); } 
    // CHECK(CNNL_EXECUTE_FUNC_S(cnnlCreate, &handle));
    cnnl_handle_.set(
            std::unique_ptr<cnnlHandle_t, void (*)(cnnlHandle_t *)>(
                    new cnnlHandle_t(handle), [](cnnlHandle_t *h) {
                        if (h != nullptr){
                            // cnnlStatus_t err = cnnlDestroy(*h);
                            // if (err != CNNL_STATUS_SUCCESS) { return cnnl_to_dnnl_status(err); }
                            CNNL_EXECUTE_FUNC_V(cnnlDestroy, *h);
                        }
                        delete h;
                    }));
    handle = nullptr;
    return status::success;
}

CNcontext sycl_bang_engine_t::get_underlying_context() const {
    return cl::sycl::get_native<cl::sycl::backend::cnrt>(context());
}

status_t sycl_bang_engine_t::create_stream(stream_t **stream, unsigned flags) {
    return sycl_bang_stream_t::create_stream(stream, this, flags);
}

status_t sycl_bang_engine_t::create_stream(
        stream_t **stream, cl::sycl::queue &queue) {
    return sycl_bang_stream_t::create_stream(stream, this, queue);
}

status_t sycl_bang_engine_t::underlying_context_type() {
    // this is a costly function which take avarage up to 75ms
    // on titanrx. So we must run it once and store the variable
    // in  is_primary_context_;
    CNcontext primary;
    CNdev device_current;   // both driver and runtime api use uint64 as device type , TODO: use driver api
    CNcontext desired = compat::get_native<CNcontext>(context());
    CNdev bang_device = compat::get_native<CNdev>(device());

    // TODO: verify if this is right
    // in cnrt, a (driver api)CNcontext associated with a CPU thread, and (runtime api)cnrtRuntimeContext_t associated device
    // in CUDA, there is no runtime context(for user), only exsist (driver api)CUcontext
    CNresult ret = cnCtxGetCurrent(&primary);
    if(ret != CN_SUCCESS){
        printf("%s@%d return %d FAILED\n",__func__, __LINE__,ret);
    }
    CNresult ret2 = cnCtxGetDevice(&device_current);
    if(ret2 != CN_SUCCESS){
        printf("%s@%d return %d FAILED\n",__func__, __LINE__,ret2);
    }
    // CHECK(BANG_EXECUTE_FUNC_S(cnCtxGetCurrent, &primary));
    // CHECK(BANG_EXECUTE_FUNC_S(cnCtxGetDevice, &device_current));
    primary_context_ = primary == desired && device_current == bang_device;

    // CHECK(CUDA_EXECUTE_FUNC_S(cuDevicePrimaryCtxRetain, &primary, cuda_device));
    // CHECK(CUDA_EXECUTE_FUNC_S(cuDevicePrimaryCtxRelease, cuda_device));
    // CHECK(BANG_EXECUTE_FUNC_S(cnrtSetRuntimeContextDeviceId, primary, bang_device));
    // CHECK(BANG_EXECUTE_FUNC_S(cnrtDestroyRuntimeContext, primary));
    // primary_context_ = (primary == desired);
    return status::success;
}

cnnlHandle_t *sycl_bang_engine_t::get_cnnl_handle() {
    if (!cnnl_handle_.is_set()) set_cnnl_handle();
    return cnnl_handle_.get().get();
}

device_id_t sycl_bang_engine_t::device_id() const {
    return device_id_t(static_cast<int>(sycl::backend_t::cambricon),
            static_cast<uint64_t>(
                    cl::sycl::get_native<cl::sycl::backend::cnrt>(device())),
            static_cast<uint64_t>(0));
}

void sycl_bang_engine_t::activate_stream_cnnl(stream_t *stream) {
    bang_sycl_scoped_context_handler_t sc(*this);
    auto bang_stream = utils::downcast<sycl_bang_stream_t *>(stream);
    auto streamId = bang_stream->get_underlying_stream();
    assert(context() == bang_stream->queue().get_context());
    cnrtQueue_t current_stream_id = nullptr;
    auto cnnlHandle_t = get_cnnl_handle();
    CNNL_EXECUTE_FUNC(cnnlGetQueue, *cnnlHandle_t, &current_stream_id);
    if (current_stream_id != streamId) {
        CNNL_EXECUTE_FUNC(cnnlSetQueue, *cnnlHandle_t, streamId);
    }
}

namespace {
using namespace dnnl::impl::data_type;
#define INSTANCE(...) \
    impl_list_item_t( \
        impl_list_item_t::type_deduction_helper_t<__VA_ARGS__::pd_t>())
// clang-format off
const dnnl::impl::impl_list_item_t sycl_bang_impl_list[] = {
        // Binary
        INSTANCE(cnnl_binary_t),

        // Split
        INSTANCE(cnnl_split_t),

        // Pooling
        INSTANCE(cnnl_pooling_fwd_t),
        INSTANCE(cnnl_pooling_bwd_t),
        
        // Matmul
        INSTANCE(cnnl_matmul_t),
        
        // Softmax
        INSTANCE(cnnl_softmax_fwd_t),
        INSTANCE(cnnl_softmax_bwd_t),
        
        // BatchNorm
        INSTANCE(cnnl_batch_normalization_fwd_t),
        INSTANCE(cnnl_batch_normalization_bwd_t),

        // BatchNorm
        INSTANCE(cnnl_lrn_fwd_t),
        INSTANCE(cnnl_lrn_bwd_t),

        // Eltwise
        INSTANCE(cnnl_eltwise_fwd_t),
        INSTANCE(cnnl_eltwise_bwd_t),

        // Convolution
        INSTANCE(cnnl_convolution_fwd_t),
        INSTANCE(cnnl_convolution_bwd_data_t),
        INSTANCE(cnnl_convolution_bwd_weights_t),

        // inner_product

        // resampling

        // deconvolution

        nullptr,
};
//clang-format on
#undef INSTANCE
} // namespace
const dnnl::impl::impl_list_item_t *sycl_bang_engine_t::get_implementation_list(
        const op_desc_t *) const {
    return sycl_bang_impl_list;
}

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl