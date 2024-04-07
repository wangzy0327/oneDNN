#ifndef GPU_AMD_SYCL_HIP_ELTWISE_IMPL_HPP
#define GPU_AMD_SYCL_HIP_ELTWISE_IMPL_HPP

#include "miopen/miopen.h"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_eltwise_impl_base_t {

public:
    virtual status_t init(const eltwise_pd_t *pd) = 0;

    virtual void execute(miopenHandle_t handle, void **x, int size) const = 0;

    virtual status_t create_and_set_act_descriptor() {
        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenCreateActivationDescriptor, &act_desc_));

        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetActivationDescriptor, act_desc_,
                act_mode, alpha, beta, gamma));

        return status::success;
    }

    // Mapping between dnnl algorithm and miopen activation mode
    status_t convert_alg_kind(
            alg_kind_t alg_kind, miopenActivationMode_t *miopen_act_mode) const {
        switch (alg_kind) {
            case alg_kind::eltwise_relu:
                *miopen_act_mode = miopenActivationMode_t::miopenActivationRELU;
                break;
            case alg_kind::eltwise_bounded_relu:
                *miopen_act_mode = miopenActivationMode_t::miopenActivationCLIPPEDRELU;
                break;
            case alg_kind::eltwise_tanh:
                *miopen_act_mode = miopenActivationMode_t::miopenActivationTANH;
                break;
            case alg_kind::eltwise_elu:
                *miopen_act_mode = miopenActivationMode_t::miopenActivationELU;
                break;
            case alg_kind::eltwise_logistic:
                *miopen_act_mode = miopenActivationMode_t::miopenActivationLOGISTIC;
                break;
            default: return status::unimplemented;
        }
        return status::success;
    }

    virtual ~miopen_eltwise_impl_base_t() {
        if (act_desc_) {
            MIOPEN_EXECUTE_FUNC_V(miopenDestroyActivationDescriptor, act_desc_);
        }
    }
protected:
    int ndims;
    miopenActivationDescriptor_t act_desc_ = nullptr;
    miopenActivationMode_t act_mode;
    // alpha, beta and gamma are used in some activation modes in miopen,
    // see: https://rocmsoftwareplatform.github.io/MIOpen/doc/html/activation.html#miopenactivationmode-t
    float alpha = 1.0f, beta = 1.0, gamma = 1.0f;
    // a, b are used as scaling arguments.
    float a = 1.0f, b = 0;
};

struct miopen_eltwise_fwd_impl_t : public miopen_eltwise_impl_base_t {
public:
    status_t init(const eltwise_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with creating
        // miopen descriptors
        if (has_zero_dims(pd->src_md()->dims, pd->ndims())) {
            return status::success;
        }
        if (pd->ndims() > 8) { return status::invalid_arguments; }
        ndims = pd->ndims() < 4 ? 4 : pd->ndims();

        // Obtain source and destination dimensions, strides and datatype
        convert_dims(pd->src_md()->padded_dims, dims_, pd->ndims());
        convert_dims(pd->src_md()->format_desc.blocking.strides, strides_, pd->ndims());
        CHECK(convert_data_type(pd->src_md(), &data_type_));

        // Get miopen activation mode
        alg_kind_t alg = pd->desc()->alg_kind;
        auto alg_ok = convert_alg_kind(alg, &act_mode);
        if (alg_ok != status::success) { return status::unimplemented; }
        // gamma = pd->desc()->alpha;

        CHECK(create_and_set_tensor_descriptor(&tensor_desc_, data_type_, ndims, dims_, strides_));
        CHECK(create_and_set_act_descriptor());
        return status::success;
    }

    void execute(miopenHandle_t handle, void **x, int size) const override {
        // Confirm that 2 arguments were passed src and dst
        assert(size == 2);
        MIOPEN_EXECUTE_FUNC(miopenActivationForward, handle, act_desc_, &a,
                tensor_desc_, x[0], &b, tensor_desc_, x[1]);
    }
    
    // ???
    ~miopen_eltwise_fwd_impl_t() {
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, tensor_desc_);
    }

private:
    int strides_[DNNL_MAX_NDIMS];
    int dims_[DNNL_MAX_NDIMS];
    miopenDataType_t data_type_;
    miopenTensorDescriptor_t tensor_desc_;
};

struct miopen_eltwise_bwd_impl_t : public miopen_eltwise_impl_base_t {

public:
    status_t init(const eltwise_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with creating
        // miopen descriptors
        if (memory_desc_wrapper(pd->desc()->data_desc).has_zero_dim())
            return status::success;

        if (pd->ndims() > 8) { return status::invalid_arguments; }
        ndims = pd->ndims() < 4 ? 4 : pd->ndims();

        // Obtain dimension and strides for the backward eltwise operation
        convert_dims(pd->src_md()->padded_dims, dims_, pd->ndims());

        convert_dims(pd->src_md()->format_desc.blocking.strides, strides_,
                pd->ndims());

        alg_kind_t alg = pd->desc()->alg_kind;
        auto alg_ok = convert_alg_kind(alg, &act_mode);
        if (alg_ok != status::success) { return status::unimplemented; }

        // Check validity of input
        assert(pd->diff_dst_md()->data_type == pd->src_md()->data_type);
        assert(pd->diff_dst_md()->data_type == pd->diff_src_md()->data_type);

        CHECK(convert_data_type(pd->src_md(), &data_type_));

        CHECK(create_and_set_tensor_descriptor(
                &tensor_desc_src_, data_type_, ndims, dims_, strides_));
        CHECK(create_and_set_tensor_descriptor(
                &tensor_diff_desc_, data_type_, ndims, dims_, strides_));
        CHECK(create_and_set_act_descriptor());
        return status::success;
    }

    void execute(miopenHandle_t handle, void **x, int size) const override {
        // Assert that 3 arguments were passed src, diff_dst and diff_src
        assert(size == 3);
        void *dy = x[1];
        void *dx = x[2];
        MIOPEN_EXECUTE_FUNC(miopenActivationBackward, handle, act_desc_, &a,
                tensor_desc_src_, x[0], tensor_diff_desc_, dy, tensor_desc_src_,
                x[0], &b, tensor_diff_desc_, dx);
    }

    ~miopen_eltwise_bwd_impl_t() {
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, tensor_desc_src_);
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, tensor_diff_desc_);
    }

private:
    int dims_[DNNL_MAX_NDIMS];
    int strides_[DNNL_MAX_NDIMS];
    miopenTensorDescriptor_t tensor_diff_desc_;
    miopenDataType_t data_type_;
    miopenTensorDescriptor_t tensor_desc_src_;
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
