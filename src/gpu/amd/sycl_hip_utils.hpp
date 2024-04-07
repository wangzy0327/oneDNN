#ifndef GPU_AMD_SYCL_HIP_UTILS_HPP
#define GPU_AMD_SYCL_HIP_UTILS_HPP

#include <stdexcept>
#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <miopen/miopen.h>
#include <iostream>

#include "dnnl_sycl.hpp"

#include "common/engine.hpp"
#include "common/z_magic.hpp"

#include <CL/sycl/backend/hip.hpp>
#include "gpu/amd/sycl_hip_compat.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

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

bool compare_hip_devices(
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

static void convert_dims(const dnnl_dim_t *dims, int *new_dims, int n_dims,
        int adjustment_size = 4, int adjustment_value = 1) {
    convert_dnnl_dims_array(dims, new_dims, n_dims);
    for (size_t i = n_dims; i < adjustment_size; i++) {
        new_dims[i] = adjustment_value;
    }
}

static bool memory_desc_matches_nchw_vect_c(const memory_desc_t *mem_desc) {
    // Only one block is supported for second (C) dimension and the block size
    // must be 4 and the dimension has to be a multiple of block size.
    auto is_int_8 = utils::one_of(mem_desc->data_type, data_type::s8);
    auto &strides = mem_desc->format_desc.blocking.strides;
    if (is_int_8 && mem_desc->format_desc.blocking.inner_nblks == 1
            && mem_desc->format_desc.blocking.inner_idxs[0] == 1
            && mem_desc->format_desc.blocking.inner_blks[0] == 4
            && mem_desc->dims[1] % 4 == 0) {
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
static bool adjust_dim_for_dnn(
        int *dims, int n_dims, const memory_desc_t *mem_desc) {
    if (memory_desc_matches_nchw_vect_c(mem_desc)) {
        dims[n_dims] = mem_desc->format_desc.blocking.inner_blks[0];
        dims[mem_desc->format_desc.blocking.inner_idxs[0]]
                /= mem_desc->format_desc.blocking.inner_blks[0];
        return true;
    }
    return false;
}

static bool adjust_stride_for_dnn(
        int *stride, int n_dims, const memory_desc_t *mem_desc) {
    if (memory_desc_matches_nchw_vect_c(mem_desc)) {
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

// verify_format - miopen
// MIOpen currently only implements NCHW layout
static status_t verify_format(const memory_desc_t *md, bool consider_ab_as_nhwc = false) {
    const memory_desc_wrapper mem_wrapper(md);
    if (mem_wrapper.matches_one_of_tag(format_tag::ab, format_tag::abc,
                format_tag::abcd, format_tag::abcde, format_tag::abcde))
        return status::success;

    if (consider_ab_as_nhwc && mem_wrapper.matches_one_of_tag(format_tag::ab))
        return status::success;

    if (mem_wrapper.matches_one_of_tag(format_tag::acdb))
        printf("miopen only support NCHW layout for 4d tensor\n");
    return status::unimplemented;
}

static bool memory_format_ok(const memory_desc_t *mem_desc) {
    return (memory_desc_matches_nchw_vect_c(mem_desc)
            || mem_desc->format_desc.blocking.inner_nblks == 0);
}
  
// convert_data_type - miopen
static status_t convert_data_type(const memory_desc_t *mem_desc,
        miopenDataType_t *miopen_data_type, bool vectorized = true) {
    switch (mem_desc->data_type) {
        case dnnl_data_type_t::dnnl_f16:
            *miopen_data_type = miopenDataType_t::miopenHalf;
            break;
        case dnnl_data_type_t::dnnl_f32:
            *miopen_data_type = miopenDataType_t::miopenFloat;
            break;
            // CUDNN_TENSOR_NCHW_VECT_C format is only supported with tensor
            // data types CUDNN_DATA_INT8x4, CUDNN_DATA_INT8x32, and
            // CUDNN_DATA_UINT8x4. oneDNN does not support UINT8 and block size
            // of 32, hence the only valid case is CUDNN_DATA_INT8x4
            // so with miopen
        case dnnl_data_type_t::dnnl_s8:
            *miopen_data_type
                    = ((vectorized
                               && mem_desc->format_desc.blocking.inner_blks[0]
                                       == 4)
                                    ? miopenDataType_t::miopenInt8x4
                                    : miopenDataType_t::miopenInt8);
            break;
        default: return status::unimplemented;
    }
    return status::success;
}

class rocblas_error : virtual public std::runtime_error {

protected:
    const char *rocblas_error_map(rocblas_status error) {
        switch (error) {
            case rocblas_status_success: return "rocblas_status_success";

            case rocblas_status_invalid_handle:
                return "rocblas_status_invalid_handle";

            case rocblas_status_not_implemented:
                return "rocblas_status_not_implemented";

            case rocblas_status_invalid_pointer:
                return "rocblas_status_invalid_pointer";

            case rocblas_status_invalid_size:
                return "rocblas_status_invalid_size";

            case rocblas_status_memory_error:
                return "rocblas_status_memory_error";

            case rocblas_status_internal_error:
                return "rocblas_status_internal_error";

            case rocblas_status_perf_degraded:
                return "rocblas_status_perf_degraded";

            case rocblas_status_size_query_mismatch:
                return "rocblas_status_size_query_mismatch";

            case rocblas_status_size_increased:
                return "rocblas_status_size_increased";
            
            case rocblas_status_size_unchanged:
                return "rocblas_status_size_unchanged";
            
            case rocblas_status_invalid_value:
                return "rocblas_status_invalid_value";
            
            case rocblas_status_continue:
                return "rocblas_status_continue";

            default: return "<unknown>";
        }
    }

    int error_number_;

public:
    explicit rocblas_error(const std::string &message, rocblas_status result)
        : std::runtime_error(
                (message + std::string(rocblas_error_map(result)))) {
        error_number_ = static_cast<int>(result);
    }

    virtual ~rocblas_error() throw() {}

    virtual int get_error_number() const throw() { return error_number_; }
};

class hip_error : virtual public std::runtime_error {

protected:
    inline const char *hip_error_map(hipError_t result) {
        switch (result) {
            case hipSuccess: return "hipSuccess";
            // case CUDA_ERROR_NOT_PERMITTED: return "CUDA_ERROR_NOT_PERMITTED"; //没找到对应的
            case hipErrorInvalidContext:
                return "hipErrorInvalidContext";
            case hipErrorInvalidDevice: return "hipErrorInvalidDevice";
            case hipErrorInvalidValue: return "hipErrorInvalidValue";
            case hipErrorOutOfMemory: return "hipErrorOutOfMemory";
            case hipErrorLaunchOutOfResources:
                return "hipErrorLaunchOutOfResources";
            default: return "<unknown>";
        }
    }
    int error_number_;

public:
    // hip driver api和runtime api 的status返回是一个数据结构 即hipError_t
    // cuda把二者分开为 CUresult(DA) 和 cudaError_t(RA)
    explicit hip_error(const std::string &message, hipError_t result)
        : std::runtime_error((message + std::string(hip_error_map(result)))) {
        error_number_ = static_cast<int>(result);
    }

    virtual ~hip_error() throw() {}

    virtual int get_error_number() const throw() { return error_number_; }
};

class miopen_error : virtual public std::runtime_error {

protected:
    const char *miopen_error_map(miopenStatus_t error) {
        switch (error) {
            case miopenStatusSuccess: return "miopenStatusSuccess";
            default: return "miopen_error";
        }
    }

    int error_number_;

public:
    explicit miopen_error(const std::string &message, miopenStatus_t result)
        : std::runtime_error(
                (message + std::string(miopen_error_map(result)))) {
        error_number_ = static_cast<int>(result);
    }

    virtual ~miopen_error() throw() {}

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

static status_t miopen_to_dnnl_status(miopenStatus_t miopen_result) {
    switch (miopen_result) {
        case miopenStatusSuccess: return status::success;
        default: return status::runtime_error;
    }
}

static status_t rocblas_to_dnnl_status(rocblas_status roc_status) {
    switch (roc_status) {
        case rocblas_status_success: return status::success;
        default: return status::runtime_error;
    }
}

static status_t hip_to_dnnl_status(hipError_t hip_result) {
    switch (hip_result) {
        case hipSuccess: return status::success;
        default: return status::runtime_error;
    }
}

#define HIP_ERROR_LOCATION __FILE__ " : " STRINGIFY(__LINE__)

#define HIP_EXECUTE_FUNC(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != hipSuccess) { \
            throw hip_error(std::string("At :") \
                            + std::string(HIP_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err); \
        } \
    }

#define ROCBLAS_EXECUTE_FUNC(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != rocblas_status_success) { \
            throw rocblas_error(std::string("At :") \
                            + std::string(HIP_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err); \
        } \
    }

#define MIOPEN_EXECUTE_FUNC(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != miopenStatusSuccess) { \
            throw miopen_error(std::string("At :") \
                            + std::string(HIP_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err); \
        } \
    }

#define HIP_EXECUTE_FUNC_V(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != hipSuccess) { \
            std::cout << hip_error(std::string("At :") \
                            + std::string(HIP_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err) \
                                 .what() \
                      << std::endl; \
        } \
    }

#define MIOPEN_EXECUTE_FUNC_V(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != miopenStatusSuccess) { \
            std::cout << miopen_error(std::string("At :") \
                            + std::string(HIP_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err) \
                                 .what() \
                      << std::endl; \
        } \
    }

#define ROCBLAS_EXECUTE_FUNC_V(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != rocblas_status_success) { \
            std::cout << rocblas_error(std::string("At :") \
                            + std::string(HIP_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err) \
                                 .what() \
                      << std::endl; \
        } \
    }

#define MIOPEN_CHECK_V(e) \
    { \
        auto status = (e); \
        if (status != miopenStatusSuccess) { \
            std::cout << miopen_error(std::string("At :") \
                            + std::string(HIP_ERROR_LOCATION) \
                            + std::string(" : "), \
                    status) \
                                 .what() \
                      << std::endl; \
        } \
    }

#define HIP_EXECUTE_FUNC_S(name, ...) \
    [&]() { \
        auto err = name(__VA_ARGS__); \
        return hip_to_dnnl_status(err); \
    }()

#define ROCBLAS_EXECUTE_FUNC_S(name, ...) \
    [&]() { \
        auto err = name(__VA_ARGS__); \
        return rocblas_to_dnnl_status(err); \
    }()

#define MIOPEN_EXECUTE_FUNC_S(name, ...) \
    [&]() { \
        auto err = name(__VA_ARGS__); \
        if (err != miopenStatusSuccess) { return miopen_to_dnnl_status(err); } \
        return status::success; \
    }()

static status_t create_and_set_tensor_descriptor(
        miopenTensorDescriptor_t *tensor_desc, miopenDataType_t data_type,
        int ndims, int *dims, int *strides) {
    CHECK(MIOPEN_EXECUTE_FUNC_S(miopenCreateTensorDescriptor, tensor_desc));
    CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetTensorDescriptor, *tensor_desc,
            data_type, ndims, dims, strides));
    return status::success;
}

static status_t create_and_set_conv_descriptor(
        miopenConvolutionDescriptor_t *conv_desc, int ndims, int *padding,
        int *strides, int *dilation, miopenConvolutionMode_t mode) {
    CHECK(MIOPEN_EXECUTE_FUNC_S(miopenCreateConvolutionDescriptor, conv_desc));
    CHECK(MIOPEN_EXECUTE_FUNC_S(miopenInitConvolutionNdDescriptor, *conv_desc,
            ndims, padding, strides, dilation, mode));
    return status::success;
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
