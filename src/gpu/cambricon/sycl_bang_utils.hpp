#ifndef GPU_CAMBRICON_SYCL_BANG_UTILS_HPP
#define GPU_CAMBRICON_SYCL_BANG_UTILS_HPP

#include <stdexcept>

#include <cnnl.h>
//
// #include <cnmlrt.h>

#include "dnnl_sycl.hpp"

#include "common/engine.hpp"
#include "common/z_magic.hpp"

#include <CL/sycl/backend/cnrt.hpp>
#include "gpu/cambricon/sycl_bang_compat.hpp"

#include <iostream>

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

#define CTX_OUT_ACCESSOR(arg) \
    utils::downcast<sycl::sycl_buffer_memory_storage_t *>( \
            &CTX_OUT_STORAGE(arg)) \
            ->buffer() \
            .get_access<cl::sycl::access::mode::write>(cgh)

#define CTX_IN_ACCESSOR(arg) \
    utils::downcast<sycl::sycl_buffer_memory_storage_t *>( \
            &CTX_IN_STORAGE(arg)) \
            ->buffer() \
            .get_access<cl::sycl::access::mode::read>(cgh)

#define CTX_SCRATCH_ACCESSOR(arg) \
    utils::downcast<sycl::sycl_buffer_memory_storage_t *>( \
            ctx.get_scratchpad_grantor().get_memory_storage(arg).get()) \
            ->buffer() \
            .get_access<cl::sycl::access::mode::read_write>(cgh)


bool compare_bang_devices(
        const cl::sycl::device &lhs, const cl::sycl::device &rhs);

// Check if the device type matches the passed engine kind
inline status_t check_device(dnnl::impl::engine_kind_t eng_kind) {
    return (eng_kind == dnnl::impl::engine_kind::gpu
                    ? status::success
                    : status::invalid_arguments);
}

static void convert_dnnl_dims_array(
        const dnnl_dim_t *dims, int *new_dims, int n_dims) {
    for (size_t i = 0; i < n_dims; i++) {
        new_dims[i] = static_cast<int>(dims[i]);
    }
}

// conver dnnl_dim_t(long) to int type
static void convert_dims(const dnnl_dim_t *dims, int *new_dims, int n_dims,
        int adjustment_size = 4, int adjustment_value = 1) {
    convert_dnnl_dims_array(dims, new_dims, n_dims);
    for (size_t i = n_dims; i < adjustment_size; i++) {
        new_dims[i] = adjustment_value;
    }
}

// TODO: check cnnl's supportablity of int_8 NHWC tensor
static bool memory_desc_matches_nhwc_vect_c(const memory_desc_t *mem_desc) {
    // Only one block is supported for the second (C) dimension and the block size
    // must be 4 and the dimension has to be a multiple of block size.
    auto is_int_8 = utils::one_of(mem_desc->data_type, data_type::s8);
    auto &strides = mem_desc->format_desc.blocking.strides;
    if (is_int_8 && mem_desc->format_desc.blocking.inner_nblks == 1
            && mem_desc->format_desc.blocking.inner_idxs[0] == 1
            && mem_desc->format_desc.blocking.inner_blks[0] == 4
            && mem_desc->dims[3] % 4 == 0) {
        for (int d = 0; d < mem_desc->ndims - 1; ++d)
            if (strides[d] < strides[d + 1]) return false;
        return true;
    }
    return false;
}

static bool has_different_block_size(
        const memory_desc_t *src_md, const memory_desc_t *dst_md) {
    return ((src_md->format_desc.blocking.inner_nblks > 0
                    && dst_md->format_desc.blocking.inner_nblks == 0)
            || (src_md->format_desc.blocking.inner_nblks == 0
                    && dst_md->format_desc.blocking.inner_nblks > 0));
}

// adjust blocking memory 
static bool adjust_dim_for_dnn(
        int *dims, int n_dims, const memory_desc_t *mem_desc) {
    if (memory_desc_matches_nhwc_vect_c(mem_desc)) {
        dims[n_dims] = mem_desc->format_desc.blocking.inner_blks[0];
        dims[mem_desc->format_desc.blocking.inner_idxs[0]]
                /= mem_desc->format_desc.blocking.inner_blks[0];
        return true;
    }
    return false;
}

static bool adjust_stride_for_dnn(
        int *stride, int n_dims, const memory_desc_t *mem_desc) {
    if (memory_desc_matches_nhwc_vect_c(mem_desc)) {
        stride[n_dims] = mem_desc->format_desc.blocking.inner_nblks;
        return true;
    }
    return false;
}

// Check if the dimensions contain any zeros, returns true if they do.
static bool has_zero_dims(const dnnl_dim_t *dims, int n_dims) {
    for (size_t i = 0; i < n_dims; i++) {
        if (dims[i] == 0) { return true; }
    }
    return false;
}

static status_t format_to_cnnl_layout(dnnl_format_tag_t format_tag, cnnlTensorLayout_t &layout){
    switch (format_tag)
    {
    case format_tag::abcd:
        layout = CNNL_LAYOUT_NCHW;
        break;
    case format_tag::acdb:
        layout = CNNL_LAYOUT_NHWC;
        break;
    case format_tag::ab:
        layout = CNNL_LAYOUT_NC;
        break;
    default:
        layout = CNNL_LAYOUT_ARRAY;
        break;
    }
    return status::success;
}

static status_t get_format(const memory_desc_t *md, cnnlTensorLayout_t &format,
        bool consider_ab_as_nhwc = false) {
    // TODO: support more format(layout)
    const memory_desc_wrapper mem_wrapper(md);
    if (mem_wrapper.matches_one_of_tag(format_tag::acdb)) {
        format = cnnlTensorLayout_t::CNNL_LAYOUT_NHWC;
    } else if (mem_wrapper.matches_one_of_tag(format_tag::nchw)) {
        format = cnnlTensorLayout_t::CNNL_LAYOUT_NCHW;
    } else if (mem_wrapper.matches_one_of_tag(format_tag::ndhwc)) {
        format = cnnlTensorLayout_t::CNNL_LAYOUT_NDHWC;
    } else if (mem_wrapper.matches_one_of_tag(format_tag::any)) {
        format = cnnlTensorLayout_t::CNNL_LAYOUT_ARRAY;
    }
    else {
        assert(0 && "not supported format!");
        return status::unimplemented;
    }
    return status::success;
}

static bool memory_format_ok(const memory_desc_t *mem_desc) {
    return (memory_desc_matches_nhwc_vect_c(mem_desc)
            || mem_desc->format_desc.blocking.inner_nblks == 0);
}

static status_t memory_desc_matches_nchw_vect_c(const memory_desc_t *md){
    return status::unimplemented;
}

// convert_data_type - cnnl, TODO: support cnnl's blocking data type
static status_t convert_data_type(const memory_desc_t *mem_desc,
        cnnlDataType_t *cnnl_data_type, bool vectorized = true) {
    switch (mem_desc->data_type) {
        case dnnl_data_type_t::dnnl_f16:
            *cnnl_data_type = cnnlDataType_t::CNNL_DTYPE_HALF;
            break;
        case dnnl_data_type_t::dnnl_f32:
            *cnnl_data_type = cnnlDataType_t::CNNL_DTYPE_FLOAT;
            break;
            //cnnl support CNNL_DTYPE_INT32  CNNL_DTYPE_FLOAT
        case dnnl_data_type_t::dnnl_s8:
            *cnnl_data_type = cnnlDataType_t::CNNL_DTYPE_INT8;
            break;
        default: return status::unimplemented;
    }
    return status::success;
}

class cnnl_error : virtual public std::runtime_error {

protected:
    const char *cnnl_error_map(cnnlStatus_t error) {
        switch (error) {
            case CNNL_STATUS_SUCCESS: return "cnnl_status_success";

            case CNNL_STATUS_NOT_INITIALIZED:
                return "cnnl_status_not_initialized";

            case CNNL_STATUS_ALLOC_FAILED:
                return "cnnl_status_alloc_failed";

            case CNNL_STATUS_BAD_PARAM:
                return "cnnl_status_bad_param";

            case CNNL_STATUS_INTERNAL_ERROR:
                return "cnnl_status_internal_error";

            case CNNL_STATUS_ARCH_MISMATCH:
                return "cnnl_status_arch_mismatch";

            case CNNL_STATUS_EXECUTION_FAILED:
                return "cnnl_status_execution_failed";

            case CNNL_STATUS_NOT_SUPPORTED:
                return "cnnl_status_not_supported";
            
            case CNNL_STATUS_NUMERICAL_OVERFLOW:
                return "cnnl_status_numerical_overflow";

            default: return "<unknown>";
        }
    }

    int error_number_;

public:
    explicit cnnl_error(const std::string &message, cnnlStatus_t result)
        : std::runtime_error(
                (message + std::string(cnnl_error_map(result)))) {
        error_number_ = static_cast<int>(result);
    }

    virtual ~cnnl_error() throw() {}

    virtual int get_error_number() const throw() { return error_number_; }
};

class bang_error : virtual public std::runtime_error {

protected:
    inline const char *bang_error_map(CNresult result) {
        switch (result) {
            case CN_SUCCESS: return "cnSuccess";
            case CN_OPS_ERROR_NOT_PERMITTED:
                return "cnOpsErrorNotPermitted";
            case CN_CONTEXT_ERROR_INVALID: return "cnContextErrorInvalid";
            case CN_ERROR_INVALID_DEVICE: return "cnErrorInvaidDevice";
            case CN_ERROR_INVALID_VALUE: return "cnErrorInvalidValue";
            case CN_MEMORY_ERROR_OUT_OF_MEMORY:
                return "cnMemoryErrorOutOfMemory";
            case CN_INVOKE_ERROR_OUT_OF_RESOURCES:
                return "cnInvokeErrorOutOfResources";
            default: return "<unknown>";
        }
    }
    int error_number_;

public:
    explicit bang_error(const std::string &message, CNresult result)
        : std::runtime_error((message + std::string(bang_error_map(result)))) {
        error_number_ = static_cast<int>(result);
    }

    virtual ~bang_error() throw() {}

    virtual int get_error_number() const throw() { return error_number_; }
};


template <typename T>
cl::sycl::event copy(cl::sycl::queue &q, T *src, cl::sycl::buffer<T, 1> &dst) {

    auto event = q.submit([&, src](cl::sycl::handler &cgh) {
        // Retrieve a  write accessor to a global buffer, deprecated, TODO change
        auto acc = dst.template get_access<::sycl::access::mode::write,
                sycl::compat::target_device>(cgh);
        // Copy from the input pointer into the buffer associated with the
        // accessor
        cgh.copy(src, acc);
    });
    return event;
}

template <typename T>
cl::sycl::event copy(cl::sycl::queue &q, cl::sycl::buffer<T, 1> &src, T *dst) {

    auto event = q.submit([&, dst](cl::sycl::handler &cgh) {
        // Retrieve a read accessor to a global buffer
        auto acc = src.template get_access<::sycl::access::mode::read,
                sycl::compat::target_device>(cgh);
        // Copy from the buffer associated with the accessor into the output
        // pointer
        cgh.copy(acc, dst);
    });

    return event;
}

template <typename T>
cl::sycl::event copy(cl::sycl::queue &q, cl::sycl::buffer<T, 1> &src,
        cl::sycl::buffer<T, 1> &dst) {
    auto event = q.submit([&](cl::sycl::handler &cgh) {
        auto src_acc
                = src.template get_access<cl::sycl::access::mode::read_write>(
                        cgh);
        auto dst_acc
                = dst.template get_access<cl::sycl::access::mode::read_write>(
                        cgh);
        cgh.copy(src_acc, dst_acc);
    });
    return event;
}

static status_t cnnl_to_dnnl_status(cnnlStatus_t cnnl_result) {
    switch (cnnl_result) {
        case CNNL_STATUS_SUCCESS: return status::success;
        default: return status::runtime_error;
    }
}

static status_t bang_to_dnnl_status(CNresult bang_result) {
    switch (bang_result) {
        case CN_SUCCESS: return status::success;
        default: return status::runtime_error;
    }
}

#define BANG_ERROR_LOCATION __FILE__ " : " STRINGIFY(__LINE__)

//cnrtSyncDevice();
#define BANG_EXECUTE_FUNC(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != CN_SUCCESS) { \
            throw bang_error(std::string("At :") \
                            + std::string(BANG_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err); \
        } \
    }

// Although stupid, syncdevice is necessary...
// cnrtSyncDevice();
#define CNNL_EXECUTE_FUNC(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != CNNL_STATUS_SUCCESS) { \
            throw cnnl_error(std::string("At:") \
                            + std::string(BANG_ERROR_LOCATION) \
                            + std::string("\n") \
                            + std::string(#name) + std::string(" : "), \
                    err); \
        } \
    }

#define BANG_EXECUTE_FUNC_V(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
         \
        if (err != CN_SUCCESS) { \
            std::cout << bang_error(std::string("At :") \
                            + std::string(BANG_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err) \
                                 .what() \
                      << std::endl; \
        } \
    }

// Although stupid, syncdevice is necessary...
// cnrtSyncDevice();
#define CNNL_EXECUTE_FUNC_V(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != CNNL_STATUS_SUCCESS) { \
            std::cout << cnnl_error(std::string("At :") \
                            + std::string(BANG_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err) \
                                 .what() \
                      << std::endl; \
            assert(0); \
        } \
    }

#define CNNL_CHECK_V(e) \
    { \
        auto status = (e); \
        if (status != CNNL_STATUS_SUCCESS) { \
            std::cout << cnnl_error(std::string("At :") \
                            + std::string(BANG_ERROR_LOCATION) \
                            + std::string(" : "), \
                    status) \
                                 .what() \
                      << std::endl; \
        } \
    }

// cnrtSyncDevice(); 
#define BANG_EXECUTE_FUNC_S(name, ...) \
    [&]() { \
        auto err = name(__VA_ARGS__); \
        return bang_to_dnnl_status(err); \
    }()

// Although stupid, syncdevice is necessary...
        // cnrtSyncDevice(); 
#define CNNL_EXECUTE_FUNC_S(name, ...) \
    [&]() { \
        auto err = name(__VA_ARGS__); \
        if (err != CNNL_STATUS_SUCCESS) { return cnnl_to_dnnl_status(err); } \
        return status::success; \
    }()

static status_t create_and_set_tensor_descriptor(
        cnnlTensorDescriptor_t *tensor_desc, cnnlTensorLayout_t layout, cnnlDataType_t data_type,
        int ndims, int *dims) {
    CHECK(CNNL_EXECUTE_FUNC_S(cnnlCreateTensorDescriptor, tensor_desc));
    CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetTensorDescriptor, *tensor_desc, layout, data_type, ndims, dims));
    return status::success;
}

static status_t create_and_set_conv_descriptor(
        cnnlConvolutionDescriptor_t *conv_desc, int ndims, int *padding,
        int *strides, int *dilation, int group_count,
        cnnlDataType_t data_type) {
    CHECK(CNNL_EXECUTE_FUNC_S(cnnlCreateConvolutionDescriptor, conv_desc));
    CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetConvolutionDescriptor, *conv_desc,
            ndims, padding, strides, dilation, group_count, data_type));
    return status::success;
}

static void quantize_array(cnnlHandle_t handle, cnnlTensorDescriptor_t tensor_desc, void* _tensor, int bitwidth, void* workspace, size_t workspace_size, 
    cnnlTensorDescriptor_t q_tensor_desc, void* q_tensor){
    void *d_position, *d_scale, *d_offset;
    cnrtMalloc(&d_position, sizeof(int));
    cnrtMalloc(&d_scale, sizeof(float));
    cnrtMalloc(&d_offset, sizeof(int));

    auto err1 = cnnlQuantizeParam(handle, CNNL_QUANTIZE_POSITION_SCALE, tensor_desc, _tensor,
        bitwidth, workspace, workspace_size, d_position, d_scale, d_offset);
    if(err1 != CNNL_STATUS_SUCCESS)
        assert(0 && "err1 at quantize_array");

    cnrtSyncDevice();

    // Have to copy these paramters to host memory
    int position;
    float scale;    
    cnrtMemcpy(&position, d_position, sizeof(int), cnrtMemcpyDevToHost);
    cnrtMemcpy(&scale, d_scale, sizeof(float), cnrtMemcpyDevToHost);
    if(scale>=2){
        // printf("scale %f greater than 2, reseted.\n", scale);
        scale = 1.99999;
    }
    if(scale<=0 || scale!=scale){
        // if scale less than 0 or is NaN, set it manually
        // printf("scale %f less than 0, reseted.\n", scale);
        scale = 0.00001;
    }
    assert(scale>0);
    
    cnrtSyncDevice();
    // printf("quantize pos:%d, scale:%f\n", position, scale);
    auto err2 = cnnlSetTensorDescriptorPositionAndScale(q_tensor_desc, position, scale);
    if(err2 != CNNL_STATUS_SUCCESS)
        assert(0 && "err2 at quantize_array");
    cnrtSyncDevice();
    
    auto err3 = cnnlQuantizeV1(handle, CNNL_QUANTIZE_POSITION_SCALE, tensor_desc, _tensor, q_tensor_desc, q_tensor);
    if(err3 != CNNL_STATUS_SUCCESS)
        assert(0 && "err3 at quantize_array");
    // CNNL_EXECUTE_FUNC(cnnlQuantizeV1, handle, CNNL_QUANTIZE_POSITION_SCALE, tensor_desc, _tensor, q_tensor_desc, q_tensor);

    cnrtSyncDevice();

    cnrtFree(d_position);
    cnrtFree(d_scale);
    cnrtFree(d_offset);
}

// static void printf_tensor(int16_t* d_array, int length, std::string log_path)
// {
//     std::vector<int16_t> host_buffer(length);
//     cnrtMemcpy(host_buffer.data(), d_array, sizeof(int16_t)*length, cnrtMemcpyDevToHost);
//     cnrtSyncDevice();
//     FILE *map_fp = fopen(log_path.data(), "w+");
//     int flag = 0;
//     while(true)
//     {
//         for(int i=0; i<256; i++)
//         {
//             if(flag<length)
//             {    
//                 fprintf(map_fp, "%d, ", host_buffer[flag]);
//                 flag ++;
//             }
//             else
//             {
//                 break;
//             }
//         }
//         if(flag >= length)
//             break;
//         else
//             fprintf(map_fp, "\n");
//     }
//     fclose(map_fp);
// }
// static void printf_tensor_f(float* map_tensor, int length, std::string log_path)
// {
//     std::vector<float> host_buffer(length);
//     cnrtMemcpy(host_buffer.data(), map_tensor, sizeof(float)*length, cnrtMemcpyDevToHost);
//     cnrtSyncDevice();
//     FILE *map_fp = fopen(log_path.data(), "w+");
//     int flag = 0;
//     while(true)
//     {
//         for(int i=0; i<512; i++)
//         {
//             if(flag<length)
//             {    
//                 fprintf(map_fp, "%f, ", host_buffer[flag]);
//                 flag ++;
//             }
//             else
//             {
//                 break;
//             }
//         }
//         if(flag >= length)
//             break;
//         else
//             fprintf(map_fp, "\n");
//     }
//     fclose(map_fp);
// }

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
