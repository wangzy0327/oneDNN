#ifndef GPU_CAMBRICON_CNNL_BINARY_IMPL_HPP
#define GPU_CAMBRICON_CNNL_BINARY_IMPL_HPP

#include "cnnl.h"
#include <iostream>
#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_binary_impl_base_t {
    enum io { src_0 = 0, src_1, dst_0, NUM_IO };
    cnnlDataType_t data_types[NUM_IO];
    int ndims;
    int dims[NUM_IO][DNNL_MAX_NDIMS];
    cnnlOpTensorDescriptor_t op_desc = nullptr;
    cnnlTensorDescriptor_t tensor_descs[NUM_IO] = {};
    cnnlOpTensorDesc_t alg_kind;
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    float alpha[2];
    float beta = 0.0f;
    void *workspace = NULL;
    size_t workspace_size = 0;

    virtual ~cnnl_binary_impl_base_t() {
        if (op_desc) {
            CNNL_EXECUTE_FUNC_V(cnnlDestroyOpTensorDescriptor, op_desc);
        }
        for (size_t i = 0; i < NUM_IO; i++) {
            if (tensor_descs[i]) {
                CNNL_EXECUTE_FUNC_V(
                        cnnlDestroyTensorDescriptor, tensor_descs[i]);
            }
        }
    }

    virtual status_t init(const binary_pd_t *pd) = 0;

    void execute(cnnlHandle_t handle, void *a, void *b, void *c) const {
        CNNL_EXECUTE_FUNC(cnnlOpTensor, handle, op_desc, &alpha[0],
                tensor_descs[src_0], a, &alpha[1], tensor_descs[src_1], b, workspace, workspace_size,
                &beta, tensor_descs[dst_0], c);
    }

    virtual status_t create_and_set_op_descriptor() {
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlCreateOpTensorDescriptor, &op_desc));

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetOpTensorDescriptor, op_desc,
                alg_kind, cnnlDataType_t::CNNL_DTYPE_FLOAT,
                cnnlNanPropagation_t::CNNL_NOT_PROPAGATE_NAN));

        return status::success;
    }

    status_t convert_alg_kind(
            alg_kind_t alg_kind, cnnlOpTensorDesc_t *cnnl_alg_kind) const {
        switch (alg_kind) {
            case alg_kind::binary_add:
                *cnnl_alg_kind = cnnlOpTensorDesc_t::CNNL_OP_TENSOR_ADD;
                break;
            case alg_kind::binary_mul:
                *cnnl_alg_kind = cnnlOpTensorDesc_t::CNNL_OP_TENSOR_MUL;
                break;
            case alg_kind::binary_sub:
                *cnnl_alg_kind = cnnlOpTensorDesc_t::CNNL_OP_TENSOR_SUB;
                break;
            default: return status::unimplemented;
        }
        return status::success;
    }
};

struct cnnl_binary_impl_t : public cnnl_binary_impl_base_t {
    int strides[NUM_IO][DNNL_MAX_NDIMS];

    status_t init(const binary_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with creating
        // cnnl descriptors
        if (has_zero_dims(pd->src_md(0)->dims, pd->ndims())) {
            return status::success;
        }
        if (pd->ndims() > CNNL_DIM_MAX) { return status::invalid_arguments; }
        ndims = pd->ndims() < 4 ? 4 : pd->ndims();
        convert_dims(pd->src_md(0)->padded_dims, dims[src_0], pd->ndims());
        convert_dims(pd->src_md(1)->padded_dims, dims[src_1], pd->ndims());
        convert_dims(pd->dst_md()->padded_dims, dims[dst_0], pd->ndims());

        convert_dims(pd->src_md(0)->format_desc.blocking.strides,
                strides[src_0], pd->ndims());
        convert_dims(pd->src_md(1)->format_desc.blocking.strides,
                strides[src_1], pd->ndims());
        convert_dims(pd->dst_md()->format_desc.blocking.strides, strides[dst_0],
                pd->ndims());
        alg_kind_t alg = pd->desc()->alg_kind;
        auto alg_ok = convert_alg_kind(alg, &alg_kind);
        if (alg_ok != status::success) { return status::unimplemented; }

        CHECK(convert_data_type(pd->src_md(0), &data_types[src_0]));
        CHECK(convert_data_type(pd->src_md(1), &data_types[src_1]));
        CHECK(convert_data_type(pd->dst_md(), &data_types[dst_0]));

        alpha[0] = pd->attr()->scales_.get(DNNL_ARG_SRC_0).scales_[0];
        alpha[1] = pd->attr()->scales_.get(DNNL_ARG_SRC_1).scales_[0];

        CHECK(create_and_set_tensor_descriptor(&tensor_descs[src_0], layout,
                data_types[src_0], ndims, dims[src_0]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[src_1], layout,
                data_types[src_1], ndims, dims[src_1]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[dst_0], layout,
                data_types[dst_0], ndims, dims[dst_0]));
        CHECK(create_and_set_op_descriptor());
        return status::success;
    }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
