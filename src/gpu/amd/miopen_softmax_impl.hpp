#ifndef GPU_AMD_MIOPEN_SOFTMAX_IMPL_HPP
#define GPU_AMD_MIOPEN_SOFTMAX_IMPL_HPP

#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_softmax_impl_base_t {
    miopenDataType_t data_type;
    int ndims;
    miopenSoftmaxAlgorithm_t alg_kind;
    // cuDNN only supports softmax on channel dimension, not miopen...
    // TODO: add another mode
    miopenSoftmaxMode_t mode = MIOPEN_SOFTMAX_MODE_CHANNEL;
    // oneDNN softmax primitive doesn't support any post-ops or attributes,
    // hence we can set alpha = 1 and beta = 0 for all cases
    float alpha = 1.0f;
    float beta = 0.0f;

    virtual ~miopen_softmax_impl_base_t() {}

    virtual status_t init(const softmax_pd_t *pd) = 0;

    virtual void execute(miopenHandle_t handle, void **x, int size) const = 0;

    // Mapping between dnnl algorithm and miopen softmax algorithm
    status_t convert_alg_kind(bool is_log_softmax, miopenSoftmaxAlgorithm_t *miopen_alg_kind) const {
        if (is_log_softmax) {
            *miopen_alg_kind = MIOPEN_SOFTMAX_LOG;
        } else {
            *miopen_alg_kind = MIOPEN_SOFTMAX_ACCURATE;
        }
        return status::success;
    }

    status_t convert_dims_softmax(const dims_t &orig_dims, int *modified_dims,
            int axis, int ndims, format_tag_t tag) const {
        // miopen only support nchw
        if(tag != dnnl_nchw)
            return status::unimplemented;
        
        // Initialise all dims to 1
        for (int i = 0; i < 4; i++) {
            modified_dims[i] = 1;
        }
        if (axis == 1) {
            // Copy dimensions into the new array
            int num_dims = ndims < 4 ? ndims : 4;
            for (int i = 0; i < num_dims; i++) {
                modified_dims[i] = orig_dims[i];
            }
            for (int i = 4; i < ndims; i++) {
                // accumulate all dims that bigger than 4 to dim-4
                modified_dims[3] *= orig_dims[i];   
            }
            return status::success;
        }
        switch (axis) {
            case 0:
                modified_dims[1] = orig_dims[axis];
                modified_dims[2] = orig_dims[1];
                for (int i = 2; i < ndims; i++) {
                    modified_dims[3] *= orig_dims[i];
                }
                break;
            default: {
                for (int i = 0; i < axis; i++) {
                    modified_dims[0] *= orig_dims[i];
                }
                modified_dims[1] = orig_dims[axis];
                if (axis == ndims - 1) { return status::success; }
                for (int i = axis + 1; i < ndims; i++) {
                    modified_dims[2] *= orig_dims[i];
                }
                break;
            }
        }
        return status::success;
    }

    status_t convert_tag(const memory_desc_t *md, format_tag_t &tag) const {
        // may be this need to change to...
        const memory_desc_wrapper mem_wrapper(md);
        if (mem_wrapper.matches_one_of_tag(format_tag::ba)) {
            tag = dnnl_cn;
        } else if (mem_wrapper.matches_one_of_tag(format_tag::ab,
                           format_tag::abc, format_tag::abcd, format_tag::abcde,
                           format_tag::abcdef)) {
            tag = dnnl_nchw;
        } else if (mem_wrapper.matches_one_of_tag(format_tag::acb,
                           format_tag::acdb, format_tag::acdeb)) {
            tag = dnnl_nhwc;
        } else {
            return status::unimplemented;
        }
        return status::success;
    }
};

struct miopen_softmax_fwd_impl_t : public miopen_softmax_impl_base_t {
    int dims[DNNL_MAX_NDIMS];
    int strides[DNNL_MAX_NDIMS];
    miopenTensorDescriptor_t tensor_desc;

    status_t init(const softmax_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with
        // creating cudnn descriptors
        if (has_zero_dims(pd->src_md(0)->dims, pd->ndims())) {
            return status::success;
        }

        if (pd->ndims() > 8) { return status::invalid_arguments; }
        ndims = pd->ndims() < 4 ? 4 : pd->ndims();

        format_tag_t tag;
        CHECK(convert_tag(pd->src_md(), tag));
        CHECK(convert_dims_softmax(pd->src_md()->padded_dims, dims, pd->axis(), pd->ndims(), tag));
        // TODO: need to be checked
        CHECK(convert_dims_softmax(pd->src_md()->format_desc.blocking.strides, strides, pd->axis(), pd->ndims(), tag));

        convert_alg_kind(pd->is_logsoftmax(), &alg_kind);

        assert(pd->src_md()->data_type == pd->dst_md()->data_type);

        CHECK(convert_data_type(pd->src_md(), &data_type));

        CHECK(create_and_set_tensor_descriptor(&tensor_desc, data_type, 4, dims, strides));
        return status::success;
    }

    void execute(miopenHandle_t handle, void **x, int size) const override {
        // Confirm that 2 arguments were passed, src and dst
        assert(size == 2);
        MIOPEN_EXECUTE_FUNC(miopenSoftmaxForward_V2, handle, &alpha,
                tensor_desc, x[0], &beta, tensor_desc, x[1], alg_kind, mode);
    }

    ~miopen_softmax_fwd_impl_t() {
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, tensor_desc);
    }
};

struct miopen_softmax_bwd_impl_t : public miopen_softmax_impl_base_t {
    int dims[DNNL_MAX_NDIMS];
    int dims_dst[DNNL_MAX_NDIMS];
    int strides[DNNL_MAX_NDIMS];
    int strides_dst[DNNL_MAX_NDIMS];

    miopenTensorDescriptor_t tensor_dst_desc;
    miopenTensorDescriptor_t tensor_diff_desc;

    status_t init(const softmax_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with
        // creating cudnn descriptors
        if (memory_desc_wrapper(pd->desc()->diff_desc).has_zero_dim())
            return status::success;

        if (pd->ndims() > 8) { return status::invalid_arguments; }
        ndims = pd->ndims() < 4 ? 4 : pd->ndims();

        format_tag_t tag;
        CHECK(convert_tag(pd->dst_md(), tag));
        CHECK(convert_dims_softmax(pd->dst_md()->padded_dims, dims_dst,
                pd->axis(), pd->ndims(), tag));
        CHECK(convert_dims_softmax(pd->diff_src_md()->padded_dims, dims,
                pd->axis(), pd->ndims(), tag));
        // also..
        CHECK(convert_dims_softmax(pd->diff_src_md()->format_desc.blocking.strides, strides,
                pd->axis(), pd->ndims(), tag));
        CHECK(convert_dims_softmax(pd->diff_src_md()->format_desc.blocking.strides, strides_dst,
                pd->axis(), pd->ndims(), tag));

        convert_alg_kind(pd->is_logsoftmax(), &alg_kind);

        assert(pd->diff_dst_md()->data_type == pd->dst_md()->data_type);
        assert(pd->diff_dst_md()->data_type == pd->diff_src_md()->data_type);

        CHECK(convert_data_type(pd->dst_md(), &data_type));

        CHECK(create_and_set_tensor_descriptor(
                &tensor_dst_desc, data_type, 4, dims_dst, strides));
        CHECK(create_and_set_tensor_descriptor(
                &tensor_diff_desc, data_type, 4, dims, strides_dst));
        return status::success;
    }

    void execute(miopenHandle_t handle, void **x, int size) const override {
        // Assert that 3 arguments were passed src, diff_dst and diff_src
        assert(size == 3);
        MIOPEN_EXECUTE_FUNC(miopenSoftmaxBackward_V2, handle, &alpha,
                tensor_dst_desc, x[0], tensor_diff_desc, x[1], &beta,
                tensor_diff_desc, x[2], alg_kind, mode);
    }

    ~miopen_softmax_bwd_impl_t() {
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, tensor_dst_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, tensor_diff_desc);
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
