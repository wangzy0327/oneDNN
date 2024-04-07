#ifndef GPU_CAMBRICON_CNNL_LRN_IMPL_HPP
#define GPU_CAMBRICON_CNNL_LRN_IMPL_HPP

#include "cnnl.h"

#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_lrn_impl_base_t {

    virtual ~cnnl_lrn_impl_base_t() {
        for (size_t i = 0; i < NUM_IO; i++) {
            if (tensor_descs[i]) {
                CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, tensor_descs[i]);
            }
        }
    }
    virtual status_t init(const lrn_pd_t *pd) = 0;
    virtual void execute(cnnlHandle_t handle, const std::vector<void *> &args) const = 0;

protected:
    enum io { src_idx = 0, dst_idx, d_src_idx, d_dst_idx, NUM_IO };
    cnnlDataType_t data_types[NUM_IO];
    int ndims;
    int dst_size;
    size_t ws_size = 1;
    int dims[NUM_IO][DNNL_MAX_NDIMS];
    int strides[NUM_IO][DNNL_MAX_NDIMS];
    float alpha = 1.0f;
    float beta = 0.0f;
    bool is_training;
    double lrn_alpha;
    double lrn_beta;
    double lrn_K;
    unsigned int lrn_N;
    cnnlLrnMode_t lrn_mode;
    cnnlTensorDescriptor_t tensor_descs[NUM_IO] = {};

    virtual status_t init_common(const lrn_pd_t *pd) {
        ndims = std::max(4, pd->ndims());
        if (ndims > 6) { return status::invalid_arguments; }

        const auto lrn_desc = pd->desc();
        const auto dst_wrap = memory_desc_wrapper(pd->dst_md());

        dst_size = dst_wrap.nelems();
        is_training = pd->desc()->prop_kind == prop_kind::forward_training;

        lrn_K = lrn_desc->lrn_k;
        lrn_N = lrn_desc->local_size;
        lrn_alpha = lrn_desc->lrn_alpha;
        lrn_beta = lrn_desc->lrn_beta;

        // Initialise lrn algorithm
        CHECK(convert_alg_kind(pd->desc()->alg_kind, &lrn_mode));

        // Set strides and dimensions
        convert_dims(pd->src_md()->padded_dims, dims[src_idx], pd->ndims());
        convert_dims(pd->src_md()->format_desc.blocking.strides, strides[src_idx], pd->ndims());

        // Set datatype
        CHECK(convert_data_type(pd->src_md(), &data_types[src_idx]));
        
        cnnlTensorLayout_t src_format;
        auto src_md = pd->is_fwd() ? pd->src_md() : pd->diff_src_md();
        CHECK(get_format(src_md, src_format));

        // Initialise tensor descriptor
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[src_idx], src_format,
                data_types[src_idx], ndims, dims[src_idx]));
        return status::success;
    }

    status_t convert_alg_kind(alg_kind_t alg_kind, cnnlLrnMode_t *bang_alg_kind) {
        if (alg_kind == alg_kind::lrn_across_channels) {
            *bang_alg_kind = cnnlLrnMode_t::CNNL_LRN_CROSS_CHANNEL;
        } else {
            // in cnnl_lrn.hpp, we checked algo_kind, only 2 algo will pass
            *bang_alg_kind = cnnlLrnMode_t::CNNL_LRN_WITHIN_CHANNEL;
        }
        return status::success;
    }
};

struct cnnl_lrn_fwd_impl_t : public cnnl_lrn_impl_base_t {

    status_t init(const lrn_pd_t *pd) override {
        CHECK(init_common(pd));

        convert_dims(pd->dst_md()->padded_dims, dims[dst_idx], pd->ndims());
        convert_dims(pd->dst_md()->format_desc.blocking.strides,
                strides[dst_idx], pd->ndims());

        CHECK(convert_data_type(pd->dst_md(), &data_types[dst_idx]));
        
		cnnlTensorLayout_t dst_format;
        auto dst_md = pd->dst_md();
        CHECK(get_format(dst_md, dst_format));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs[dst_idx], dst_format,
                data_types[dst_idx], ndims, dims[dst_idx]));

		// check workspace 
		int ws_ndims = pd->workspace_md()->ndims;
		auto ws_dims = pd->workspace_md()->dims;
		for(int i=0; i<ws_ndims; i++){
			ws_size *= ws_dims[i];
		}

        return status::success;
    }

    void execute(cnnlHandle_t handle,
            const std::vector<void *> &args) const override {
		auto x = args[0]; auto y = args[1]; auto ws = args[2];
		
		size_t quried_ws_size = 0;
		CNNL_EXECUTE_FUNC(cnnlGetLrnWorkspaceSize_v2, handle, tensor_descs[src_idx], tensor_descs[dst_idx], lrn_mode, lrn_N, &quried_ws_size);
        if(quried_ws_size > ws_size)
		{
			// printf("workspace size not enough\n");
            void* extra_ws;
            cnrtMalloc(&extra_ws, quried_ws_size);
            cnrtSyncDevice();

		    CNNL_EXECUTE_FUNC(cnnlLrn, handle, lrn_mode, lrn_N, lrn_alpha, lrn_beta, lrn_K,
					ws, ws_size, tensor_descs[src_idx], x, tensor_descs[dst_idx], y);
            cnrtSyncDevice();
            cnrtFree(extra_ws);
		}
        else
        {
            CNNL_EXECUTE_FUNC(cnnlLrn, handle, lrn_mode, lrn_N, lrn_alpha, lrn_beta, lrn_K,
                        ws, ws_size, tensor_descs[src_idx], x, tensor_descs[dst_idx], y);
        }
    }
};

struct cnnl_lrn_bwd_impl_t : public cnnl_lrn_impl_base_t {

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

		cnnlTensorLayout_t diff_src_format, diff_dst_format;
		auto diff_src_md = pd->diff_src_md();
		auto diff_dst_md = pd->diff_dst_md();
		CHECK(get_format(diff_src_md, diff_src_format));
        CHECK(get_format(diff_dst_md, diff_dst_format));

        // Initialise tensor descriptors
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[dst_idx], diff_dst_format,
                data_types[dst_idx], ndims, dims[dst_idx]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[d_src_idx], diff_src_format,
                data_types[d_src_idx], ndims, dims[d_src_idx]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs[d_dst_idx], diff_dst_format,
                data_types[d_dst_idx], ndims, dims[d_dst_idx]));
        return status::success;
    }

    void execute(cnnlHandle_t handle,
            const std::vector<void *> &args) const override {

        CNNL_EXECUTE_FUNC_V(cnnlLrnGrad, handle, lrn_mode, lrn_N, lrn_alpha,
                lrn_beta, lrn_K, tensor_descs[src_idx], args[src_idx],
                tensor_descs[d_dst_idx], args[d_dst_idx],
                tensor_descs[d_src_idx], args[d_src_idx]);
    }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
