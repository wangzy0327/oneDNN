#ifndef GPU_AMD_MIOPEN_CONV_INNER_PRODUCT_IMPL_HPP
#define GPU_AMD_MIOPEN_CONV_INNER_PRODUCT_IMPL_HPP

#include "miopen/miopen.h"

#include "common/type_helpers.hpp"
#include "gpu/amd/miopen_conv_filter_adjustment_base.hpp"
#include "gpu/amd/miopen_inner_product_impl.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_conv_inner_product_impl_base_t
    : public miopen_inner_product_impl_base_t,
      public miopen_conv_filter_adjustment_base_t {

    bool unfold_dimensions_ = false;
    miopenConvolutionDescriptor_t conv_desc_ = nullptr;

	// MIOPEN have limited support for convolution
    status_t filter_tag(const memory_desc_t &md, format_tag_t &weight_tag) const {
        using namespace format_tag;
        weight_tag = memory_desc_matches_one_of_tag(md, oidhw,
                oihw, oiw, aBcd4b, any); // blocked layouts
        if (weight_tag == undef) return status::unimplemented;
        return status::success;
    }
	// MIOPEN have limited support for convolution
    status_t source_tag(const memory_desc_t &md, format_tag_t &src_tag) const {
        using namespace format_tag;
        src_tag = memory_desc_matches_one_of_tag(md, ncdhw, nchw, ncw, aBcd4b, any);
        if (src_tag == undef) return status::unimplemented;
        return status::success;
    }

    virtual ~miopen_conv_inner_product_impl_base_t() {
        if (conv_desc_) {
            MIOPEN_EXECUTE_FUNC_V(miopenDestroyConvolutionDescriptor, conv_desc_);
        }
        if (tensor_descs_[io::wei]) {
            MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, tensor_descs_[io::wei]);
        }
        for (size_t i = 0; i < NUM_IO - 1; i++) {
            if (tensor_descs_[i]) {
                MIOPEN_EXECUTE_FUNC_V(
                        miopenDestroyTensorDescriptor, tensor_descs_[i]);
            }
        }
    }

	// n dim tensor -> 2 dim tensor
    void unfold_dims(io memory_index, int *folded_dims, int *folded_strides, int ndims) {
        folded_dims[0] = dims_[memory_index][0];
        folded_dims[1] = dims_[memory_index][1];
        for (int i = 2; i < ndims; i++) {
            folded_dims[1] *= dims_[memory_index][i];
            folded_dims[i] = 1;
        }
        for (int i = 2; i < ndims; i++) {
            folded_strides[i] = 1;
        }

        folded_strides[1] = 1;
        folded_strides[0] = folded_dims[1];
    }

    virtual void execute(miopenHandle_t handle, rocblas_handle, const std::vector<void *> &args) const = 0;
};

struct miopen_conv_inner_product_fwd_impl_t
    : public miopen_conv_inner_product_impl_base_t {
    bool use_fused_path_for_blocking_ = false;
    bool input_is_blocked_ = false;
    bool filter_is_blocked_ = false;
	uint64_t solution_id_;

    ~miopen_conv_inner_product_fwd_impl_t() {}
    virtual status_t init(engine_t *engine, inner_product_pd_t *pd,
            bool with_relu, bool with_eltwise, bool with_sum,
            bool use_fuse_path_for_blocking) override {
        with_bias_ = pd->with_bias();
        with_relu_ = with_relu;
        with_eltwise_ = with_eltwise;
        use_fused_path_for_blocking_ = use_fuse_path_for_blocking;
        output_scales_ = pd->attr()->output_scales_.scales_[0];
        with_sum_ = with_sum;
        scale_bias_ = (output_scales_ != 1) && with_bias_;
        // scaling factor to add the previous destination value to the current
        // computation
        sum_scale_ = sum_scale(pd);
        input_is_blocked_ = pd->src_md()->format_desc.blocking.inner_blks[0] == 4;
        filter_is_blocked_ = pd->weights_md(0)->format_desc.blocking.inner_blks[0] == 4;
        // Pad out the dimensions to at least 4.
        if (pd->ndims() > 8 || pd->ndims() < 2) {
            return status::invalid_arguments;
        }
        ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();
        // Initialise meta-data from the descriptors.
        // Convert the padded dimensions to the dimensions expected by miopen.
        get_4d_tensor_descriptor(pd->src_md(), dims_[io::src], strides_[io::src]);
        get_4d_tensor_descriptor(pd->weights_md(), dims_[io::wei], strides_[io::wei]);
        get_4d_tensor_descriptor(pd->dst_md(), dims_[io::dst], strides_[io::dst]);

        // Convert oneDNN data types to their miopen counterparts.
        CHECK(convert_data_type(pd->src_md(), &data_types_[io::src]));
        CHECK(convert_data_type(pd->weights_md(0), &data_types_[io::wei]));
        if (input_is_blocked_) {
            data_types_[io::dst] = miopenInt8x4;
        } else {
            CHECK(convert_data_type(pd->dst_md(), &data_types_[io::dst]));
        }

        // Ensure INT8 types are accumulated with INT32.
        if (data_types_[io::src] != miopenHalf
                && data_types_[io::src] != miopenFloat) {
            data_types_[NUM_IO] = miopenInt32;
        }

        format_tag_t w_tag, s_tag;
        CHECK(filter_tag(*pd->weights_md(0), w_tag));
        CHECK(source_tag(*pd->src_md(0), s_tag));
        CHECK(verify_format(pd->src_md(), pd->src_md()->ndims == 2));

        if (scale_bias_) {
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_adjusted_scales,
                    memory_desc_wrapper(pd->weights_md(1)).size(), size_t(1));
        }

        // Copy over the strides.
        if (with_bias_) {
            CHECK(convert_data_type(pd->weights_md(1), &data_types_[io::bia]));
            set_bias_dims(ndims_, pd->OC());
        }

        // source format and weight format are the same at this stage
        if (unfold_dimensions_) {
            unfold_dims(io::wei, dims_[io::wei], strides_[io::wei], ndims_);
            unfold_dims(io::src, dims_[io::src], strides_[io::src], ndims_);
            ndims_ = 4;
        }

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::src],
                data_types_[io::src], ndims_, dims_[io::src],
                strides_[io::src]));

        if (with_bias_) {
            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::bia],
                    data_types_[io::bia], ndims_, dims_[io::bia],
                    strides_[io::bia]));
        }

		CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::dst],
				data_types_[io::dst], ndims_, dims_[io::dst], 
				strides_[io::dst]));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::wei],
                data_types_[io::wei], ndims_, dims_[io::wei],
                strides_[io::wei]));

        // Set the convolution. For inner product, this means unit strides and
        // dilation, no padding, and with cross-correlation as the mode.
        int conv_dims = ndims_ - 2;
        std::vector<int> unit_strides(conv_dims, 1);
        std::vector<int> unit_dilation(conv_dims, 1);
        std::vector<int> zero_padding(conv_dims, 0);

        CHECK(create_and_set_conv_descriptor(&conv_desc_, conv_dims,
                zero_padding.data(), unit_strides.data(), unit_dilation.data(),
                miopenConvolutionMode_t::miopenConvolution));

        auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto hip_stream = utils::downcast<sycl_hip_stream_t *>(service_stream);
        auto handle = hip_stream->get_miopen_handle();

		size_t requested_algo_count, returned_algo_count;
		CHECK(MIOPEN_EXECUTE_FUNC_S(
				miopenConvolutionForwardGetSolutionCount, handle,
				tensor_descs_[io::wei], tensor_descs_[io::src], conv_desc_,
				tensor_descs_[io::dst], &requested_algo_count));
		std::vector<miopenConvSolution_t> perf_results(requested_algo_count);
		CHECK(MIOPEN_EXECUTE_FUNC_S(miopenConvolutionForwardGetSolution,
				handle, tensor_descs_[io::wei], tensor_descs_[io::src],
				conv_desc_, tensor_descs_[io::dst], requested_algo_count,
				&returned_algo_count, perf_results.data()));

		// perf_results are sorted ascending by compute time, so the first suitable
		// algorithm found is the one with best performance
		if(returned_algo_count >= 1){
            solution_id_ = perf_results[0].solution_id;
            workspace_size_ = perf_results[0].workspace_size;
        }
			
		else
			return status::invalid_arguments;

        // Allocate the workspace from the algorithm selection, if applicable.
        // CHECK(MIOPEN_EXECUTE_FUNC_S(miopenConvolutionForwardGetSolutionWorkspaceSize,
        //         handle, tensor_descs_[io::wei], tensor_descs_[io::src], conv_desc_,
        //         tensor_descs_[io::dst], solution_id_, &workspace_size_));

        if (workspace_size_ > 0) {
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt,
                    workspace_size_, size_t(1));
        }

        // Add the eltwise op. Note that this only applies to the forward pass.
        // CHECK(create_and_set_op_descriptor(pd));
        return status::success;
    }

    void execute(miopenHandle_t handle, rocblas_handle, const std::vector<void *> &args) const override {
        auto x = args[0], w = args[1], b = args[2], y = args[3],
             workspace = args[4];
        assert(args.size() == 7);
        auto w_arg = w;

        if (filter_using_spatial_format_) {
            // void *transformed_w = args[5];
            // transform_filter(handle, w, transformed_w);
            // w_arg = transformed_w;
        }

		MIOPEN_EXECUTE_FUNC_V(miopenConvolutionForwardImmediate, handle, 
			tensor_descs_[io::wei], w_arg, tensor_descs_[io::src], x, conv_desc_, tensor_descs_[io::dst], y, 
			workspace, workspace_size_, solution_id_);
      
		if (with_bias_) {
			MIOPEN_EXECUTE_FUNC_V(miopenConvolutionForwardBias, handle, &alpha_,
				tensor_descs_[io::bia], b, &beta_, tensor_descs_[io::dst], y);
		}
    }
};

struct miopen_conv_inner_product_bwd_data_impl_t
    : public miopen_conv_inner_product_impl_base_t {
    uint64_t solution_id_;
    virtual status_t init(engine_t *engine, inner_product_pd_t *pd,
            bool /*with_relu*/, bool /*with_eltwise*/, bool /*with_sum */,
            bool /*using_fused_path_for_blocking*/) override {
        // Pad out the dimensions to 4
        if (pd->ndims() > 8 || pd->ndims() < 2) {
            return status::invalid_arguments;
        }
        ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();
        // Initialise meta-data from the descriptors.
        // Convert the padded dimensions to the dimensions expected by miopen.
        get_4d_tensor_descriptor(pd->diff_src_md(), dims_[io::src], strides_[io::src]);
        get_4d_tensor_descriptor(pd->weights_md(), dims_[io::wei], strides_[io::wei]);
        get_4d_tensor_descriptor(pd->diff_dst_md(), dims_[io::dst], strides_[io::dst]);

        // Convert oneDNN data types to their miopen counterparts.
        CHECK(convert_data_type(pd->diff_src_md(), &data_types_[io::src]));
        CHECK(convert_data_type(pd->weights_md(0), &data_types_[io::wei]));
        CHECK(convert_data_type(pd->diff_dst_md(), &data_types_[io::dst]));

        format_tag_t w_tag, s_tag;
        CHECK(filter_tag(*pd->weights_md(0), w_tag));
        CHECK(source_tag(*pd->diff_src_md(0), s_tag));
        
		CHECK(verify_format(pd->diff_src_md()));
		CHECK(verify_format(pd->weights_md(0)));

        // Set the tensor descriptors from the dimensions and strides.
        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::src],
                data_types_[io::src], ndims_, dims_[io::src],
                strides_[io::src]));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::dst],
                data_types_[io::dst], ndims_, dims_[io::dst],
                strides_[io::dst]));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::wei], 
                data_types_[io::wei], ndims_, dims_[io::wei],
                strides_[io::wei]));

        // Set the convolution. For inner product, this means unit strides and
        // dilation, no padding, and with cross-correlation as the mode.
        int conv_dims = ndims_ - 2;
        std::vector<int> unit_strides(conv_dims, 1);
        std::vector<int> unit_dilation(conv_dims, 1);
        std::vector<int> zero_padding(conv_dims, 0);

        CHECK(create_and_set_conv_descriptor(&conv_desc_, conv_dims,
                zero_padding.data(), unit_strides.data(), unit_dilation.data(),
                miopenConvolutionMode_t::miopenConvolution));

        auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto hip_stream = utils::downcast<sycl_hip_stream_t *>(service_stream);
        auto handle = hip_stream->get_miopen_handle();

        // Inner product can choose whatever algorithm it prefers.
        size_t requested_algo_count, returned_algo_count;
        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenConvolutionBackwardDataGetSolutionCount, handle,
                tensor_descs_[io::dst], tensor_descs_[io::wei], conv_desc_,
                tensor_descs_[io::src], &requested_algo_count));
        std::vector<miopenConvSolution_t> perf_results(requested_algo_count);
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenConvolutionBackwardDataGetSolution,
                handle, tensor_descs_[io::dst], tensor_descs_[io::wei],
                conv_desc_, tensor_descs_[io::src], requested_algo_count,
                &returned_algo_count, perf_results.data()));
		
		if(returned_algo_count >= 1)
			solution_id_ = perf_results[0].solution_id;
		else
			return status::invalid_arguments;
        
		// Allocate the workspace from the algorithm selection, if applicable.
        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenConvolutionBackwardDataGetWorkSpaceSize, handle,
                tensor_descs_[io::dst], tensor_descs_[io::wei], conv_desc_,
                tensor_descs_[io::src], &workspace_size_));

        if (workspace_size_ > 0) {
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt,
                    workspace_size_, size_t(1));
        }

        return status::success;
    }

    void execute(miopenHandle_t handle, rocblas_handle, const std::vector<void *> &args) const override {
        assert(args.size() == 5);
        auto dx = args[0], w = args[1], dy = args[2], workspace = args[3];
        auto w_arg = w;
        if (filter_using_spatial_format_) {
			// not useable
            auto transformed_w = args[4];
            transform_filter(handle, w, transformed_w);
            w_arg = transformed_w;
        }

        MIOPEN_EXECUTE_FUNC_V(miopenConvolutionBackwardDataImmediate, handle,
                tensor_descs_[io::dst], dy, tensor_descs_[io::wei], w_arg,
                conv_desc_, tensor_descs_[io::src], dx, workspace,
                workspace_size_, solution_id_);
    }
};

struct miopen_conv_inner_product_bwd_weights_impl_t
    : public miopen_conv_inner_product_impl_base_t {
	uint64_t solution_id_;
    virtual status_t init(engine_t *engine, inner_product_pd_t *pd,
            bool /*with_relu*/, bool /*with_eltwise*/, bool /*with_sum */,
            bool /*using_fused_path_for_blocking*/) override {
        // If any of the dimensions are 0 we should not continue with creating
        // miopen descriptors
        with_bias_ = pd->with_bias();
        // Pad out the dimensions to 4
        if (pd->ndims() > 8 || pd->ndims() < 2) {
            return status::invalid_arguments;
        }
        ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();

        // Initialise meta-data from the descriptors.
        // Convert the padded dimensions to the dimensions expected by miopen.
        get_4d_tensor_descriptor(pd->src_md(), dims_[io::src], strides_[io::src]);
        get_4d_tensor_descriptor(pd->diff_weights_md(), dims_[io::wei], strides_[io::wei]);
        get_4d_tensor_descriptor(pd->diff_dst_md(), dims_[io::dst], strides_[io::dst]);

        format_tag_t w_tag, s_tag;
        CHECK(filter_tag(*pd->diff_weights_md(0), w_tag));
        CHECK(source_tag(*pd->src_md(0), s_tag));

        CHECK(verify_format(pd->src_md(0)));
		CHECK(verify_format(pd->diff_weights_md(0)));

        // Copy over the strides.
        // Convert oneDNN data types to their miopen counterparts.
        CHECK(convert_data_type(pd->src_md(), &data_types_[io::src]));
        CHECK(convert_data_type(pd->diff_weights_md(0), &data_types_[io::wei]));
        CHECK(convert_data_type(pd->diff_dst_md(), &data_types_[io::dst]));

        // source format and weight format are the same at this stage
        if (unfold_dimensions_) {
            unfold_dims(io::wei, dims_[io::wei], strides_[io::wei], ndims_);
            unfold_dims(io::src, dims_[io::src], strides_[io::src], ndims_);
            ndims_ = 4;
        }

        if (with_bias_) {
            set_bias_dims(ndims_, pd->OC());
            CHECK(convert_data_type(pd->diff_weights_md(1), &data_types_[io::bia]));
        }
        // Set the tensor descriptors from the dimensions and strides.
        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::src],
                data_types_[io::src], ndims_, dims_[io::src],
                strides_[io::src]));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::wei],
                data_types_[io::wei], ndims_, dims_[io::wei],
                strides_[io::wei]));

        // oneDNN does not set unused dimensions and strides in the output, so
        // we do that here. If nhwc filter, then repeat the N stride for the
        // spatial dimensions.
        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::dst],
                data_types_[io::dst], ndims_, dims_[io::dst],
                strides_[io::dst]));
        if (with_bias_) {
            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::bia],
                    data_types_[io::bia], ndims_, dims_[io::bia],
                    strides_[io::bia]));
        }
        // Set the convolution. For inner product, this means unit strides and
        // dilation, no padding, and with cross-correlation as the mode.
        int conv_dims = ndims_ - 2;
        std::vector<int> unit_strides(conv_dims, 1);
        std::vector<int> unit_dilation(conv_dims, 1);
        std::vector<int> zero_padding(conv_dims, 0);

        CHECK(create_and_set_conv_descriptor(&conv_desc_, conv_dims,
                zero_padding.data(), unit_strides.data(), unit_dilation.data(),
                miopenConvolutionMode_t::miopenConvolution));
        auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto hip_stream = utils::downcast<sycl_hip_stream_t *>(service_stream);
        auto handle = hip_stream->get_miopen_handle();

        // Inner product can choose whatever algorithm it prefers.
        size_t requested_algo_count, returned_algo_count;
        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenConvolutionBackwardWeightsGetSolutionCount, handle,
                tensor_descs_[io::dst], tensor_descs_[io::src], conv_desc_,
                tensor_descs_[io::wei], &requested_algo_count));
        std::vector<miopenConvSolution_t> perf_results(requested_algo_count);
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenConvolutionBackwardWeightsGetSolution,
                handle, tensor_descs_[io::dst], tensor_descs_[io::src],
                conv_desc_, tensor_descs_[io::wei], requested_algo_count,
                &returned_algo_count, perf_results.data()));

		if(returned_algo_count >= 1)
			solution_id_ = perf_results[0].solution_id;
		else
			return status::invalid_arguments;

        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenConvolutionBackwardWeightsGetWorkSpaceSize, handle,
                tensor_descs_[io::dst], tensor_descs_[io::src], conv_desc_,
                tensor_descs_[io::wei], &workspace_size_));

        if (workspace_size_ > 0) {
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt,
                    workspace_size_, size_t(1));
        }

        return status::success;
    }

    void execute(miopenHandle_t handle, rocblas_handle, const std::vector<void *> &args) const override {
        assert(args.size() == 6);
        auto x = args[0], dy = args[1], dw = args[2], db = args[3],
             workspace = args[4];
        auto dw_arg = filter_using_spatial_format_ ? args[5] : dw;
        MIOPEN_EXECUTE_FUNC_V(miopenConvolutionBackwardWeightsImmediate, handle,
                tensor_descs_[io::dst], dy, tensor_descs_[io::src], x,
                conv_desc_, tensor_descs_[io::wei], dw_arg, workspace,
                workspace_size_, solution_id_);
        if (with_bias_) {
            MIOPEN_EXECUTE_FUNC_V(miopenConvolutionBackwardBias, handle,
                    &alpha_, tensor_descs_[io::dst], dy, &beta_,
                    tensor_descs_[io::bia], db);
        }
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
