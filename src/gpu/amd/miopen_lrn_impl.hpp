#ifndef GPU_AMD_MIOPEN_LRN_IMPL_HPP
#define GPU_AMD_MIOPEN_LRN_IMPL_HPP

#include "miopen/miopen.h"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_lrn_impl_base_t {

    virtual ~miopen_lrn_impl_base_t() {
        if (lrn_desc) {
            MIOPEN_EXECUTE_FUNC_V(miopenDestroyLRNDescriptor, lrn_desc);
        }
        for (size_t i = 0; i < NUM_IO; i++) {
            if (tensor_descs[i]) {
                MIOPEN_EXECUTE_FUNC_V(
                        miopenDestroyTensorDescriptor, tensor_descs[i]);
            }
        }
    }
    virtual status_t init(const lrn_pd_t *pd) = 0;
    virtual void execute(
            miopenHandle_t handle, const std::vector<void *> &args) const = 0;

protected:
    enum io { src_idx = 0, dst_idx, d_src_idx, d_dst_idx, NUM_IO };
    miopenDataType_t data_types[NUM_IO];
    int ndims;
    int dst_size;
    int dims[NUM_IO][DNNL_MAX_NDIMS];
    int strides[NUM_IO][DNNL_MAX_NDIMS];
    float alpha = 1.0f;
    float beta = 0.0f;
    bool is_training;
    double lrn_alpha;
    double lrn_beta;
    double lrn_K;
    unsigned int lrn_N;
    miopenLRNMode_t lrn_mode;
    miopenLRNDescriptor_t lrn_desc = nullptr;
    miopenTensorDescriptor_t tensor_descs[NUM_IO] = {};

    virtual status_t init_common(const lrn_pd_t *pd) {
        ndims = std::max(4, pd->ndims());
        if (ndims > 6) { return status::invalid_arguments; }

        const bool do_scaling
                = pd->src_md()->data_type == dnnl_data_type_t::dnnl_s8;
        const auto scales_0 = pd->attr()->scales_.get(1).scales_;
        const auto lrn_desc = pd->desc();
        const auto dst_wrap = memory_desc_wrapper(pd->dst_md());

        dst_size = dst_wrap.nelems();
        alpha = do_scaling ? scales_0[0] : 1.0f;
        is_training = pd->desc()->prop_kind == prop_kind::forward_training;

        lrn_K = lrn_desc->lrn_k;
        lrn_N = lrn_desc->local_size;
        lrn_alpha = lrn_desc->lrn_alpha;
        lrn_beta = lrn_desc->lrn_beta;

        // Initialise lrn algorithm
        CHECK(convert_alg_kind(pd->desc()->alg_kind, &lrn_mode));

        // Set strides and dimensions
        convert_dims(pd->src_md()->padded_dims, dims[src_idx], pd->ndims());
        convert_dims(pd->src_md()->format_desc.blocking.strides,
                strides[src_idx], pd->ndims());

        // Set datatype
        CHECK(convert_data_type(pd->src_md(), &data_types[src_idx]));

        // Initialise tensor descriptor
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[src_idx],
                data_types[src_idx], ndims, dims[src_idx], strides[src_idx]));
        CHECK(create_and_set_lrn_descriptor());
        return status::success;
    }

    virtual status_t create_and_set_lrn_descriptor() {
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenCreateLRNDescriptor, &lrn_desc));
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetLRNDescriptor, lrn_desc, lrn_mode, lrn_N,
                lrn_alpha, lrn_beta, lrn_K));
        return status::success;
    }

    status_t convert_alg_kind(
            alg_kind_t alg_kind, miopenLRNMode_t *mode_p) {
        if(alg_kind == alg_kind::lrn_across_channels)
            *mode_p = miopenLRNMode_t::miopenLRNCrossChannel;
        else if(alg_kind == alg_kind::lrn_within_channel)
            *mode_p = miopenLRNMode_t::miopenLRNWithinChannel;
        else
            return status::unimplemented;
        return status::success;
    }
};

struct miopen_lrn_fwd_impl_t : public miopen_lrn_impl_base_t {

    status_t init(const lrn_pd_t *pd) override {
        CHECK(init_common(pd));

        convert_dims(pd->dst_md()->padded_dims, dims[dst_idx], pd->ndims());
        convert_dims(pd->dst_md()->format_desc.blocking.strides,
                strides[dst_idx], pd->ndims());

        CHECK(convert_data_type(pd->dst_md(), &data_types[dst_idx]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[dst_idx],
                data_types[dst_idx], ndims, dims[dst_idx], strides[dst_idx]));
        return status::success;
    }

    void execute(miopenHandle_t handle,
            const std::vector<void *> &args) const override {
        // args[2] is a copy of args[dst_idx] (for backward)
        // 2 not index d_src (in forward propagate)
        MIOPEN_EXECUTE_FUNC(miopenLRNForward, handle, lrn_desc,
                &alpha, tensor_descs[src_idx], args[0], &beta,
                tensor_descs[dst_idx], args[1], is_training, args[2]);
    }
};
struct miopen_lrn_bwd_impl_t : public miopen_lrn_impl_base_t {

    status_t init(const lrn_pd_t *pd) override {
        CHECK(init_common(pd));

        // Set dimensions
        convert_dims(
                pd->diff_dst_md()->padded_dims, dims[dst_idx], pd->ndims());
        convert_dims(
                pd->diff_src_md()->padded_dims, dims[d_src_idx], pd->ndims());
        convert_dims(
                pd->diff_dst_md()->padded_dims, dims[d_dst_idx], pd->ndims());

        // Set strides
        convert_dims(pd->diff_dst_md()->format_desc.blocking.strides,
                strides[dst_idx], pd->ndims());
        convert_dims(pd->diff_src_md()->format_desc.blocking.strides,
                strides[d_src_idx], pd->ndims());
        convert_dims(pd->diff_dst_md()->format_desc.blocking.strides,
                strides[d_dst_idx], pd->ndims());

        // Set datatypes
        CHECK(convert_data_type(pd->diff_dst_md(), &data_types[dst_idx]));
        CHECK(convert_data_type(pd->diff_src_md(), &data_types[d_src_idx]));
        CHECK(convert_data_type(pd->diff_dst_md(), &data_types[d_dst_idx]));

        // Initialise tensor descriptors
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[dst_idx],
                data_types[dst_idx], ndims, dims[dst_idx], strides[dst_idx]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[d_src_idx],
                data_types[d_src_idx], ndims, dims[d_src_idx],
                strides[d_src_idx]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[d_dst_idx],
                data_types[d_dst_idx], ndims, dims[d_dst_idx],
                strides[d_dst_idx]));
        return status::success;
    }

    void execute(miopenHandle_t handle,
            const std::vector<void *> &args) const override {

        MIOPEN_EXECUTE_FUNC_V(miopenLRNBackward, handle, lrn_desc,
                &alpha, tensor_descs[dst_idx], args[dst_idx],
                tensor_descs[d_dst_idx], args[d_dst_idx], 
                tensor_descs[src_idx], args[src_idx], 
                &beta, tensor_descs[d_src_idx], args[d_src_idx], args[dst_idx]);
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
