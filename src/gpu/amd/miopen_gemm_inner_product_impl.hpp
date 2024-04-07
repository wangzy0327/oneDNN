#ifndef GPU_AMD_MIOPEN_GEMM_INNER_PRODUCT_IMPL_HPP
#define GPU_AMD_MIOPEN_GEMM_INNER_PRODUCT_IMPL_HPP

#include "common/type_helpers.hpp"
#include "gpu/amd/miopen_inner_product_impl.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

// GEMM Implementation
struct miopen_gemm_inner_product_impl_base_t {
protected:
    rocblas_int m_, n_, k_, lda_, ldb_, ldc_;
    rocblas_operation trans_a_, trans_b_;
    rocblas_datatype a_type_, b_type_, c_type_, compute_type_;
    const double zero_ = 0; 
    
    // TODO: this flags' and solution_index's effect should be confirmed   
    rocblas_gemm_flags flags_ = rocblas_gemm_flags_none;
    int32_t solution_index = 0; 
    rocblas_gemm_algo algo_ = rocblas_gemm_algo_standard; // only this algo is avilable in rocblas

    status_t get_rocblas_data_type(
            const miopenDataType_t &miopen_dt, rocblas_datatype &blas_dt) const {
        switch (miopen_dt) {
            
            case miopenHalf: blas_dt = rocblas_datatype_f16_r; return status::success;
            case miopenFloat: blas_dt = rocblas_datatype_f32_r; return status::success;
            case miopenInt32: blas_dt = rocblas_datatype_i32_r; return status::success;
            case miopenInt8: blas_dt = rocblas_datatype_i8_r; return status::success;
            case miopenInt8x4: blas_dt = rocblas_datatype_i8_r; return status::success;
            default: return status::unimplemented;
        }
        return status::unimplemented;
    }
    
    status_t set_computetype(){
        if(b_type_ != a_type_)
            return status::unimplemented;
        
        // int32 TODO: confirm this
        if(a_type_ == rocblas_datatype_i32_r)
            return status::unimplemented;
        // int8 
        if(a_type_ == rocblas_datatype_i8_r && c_type_ != rocblas_datatype_f32_r)
            return status::unimplemented;  
        // double
        if(a_type_ == rocblas_datatype_f64_r){
            // In rocblas gemm, if a's type is double, then c's type must be double too.
            if(c_type_ != a_type_)
                return status::unimplemented;
            compute_type_ = rocblas_datatype_f64_r;
        }
        // half and float
        if(c_type_ != a_type_)
            compute_type_ = c_type_;
        else
            compute_type_ = rocblas_datatype_f32_r;
        return status::success;
    }
};

struct miopen_gemm_inner_product_fwd_impl_t
    : public miopen_inner_product_impl_base_t,
      public miopen_gemm_inner_product_impl_base_t,
      public miopen_conv_filter_adjustment_base_t {

    miopenActivationDescriptor_t act_desc_;
    bool use_acc_dst_;
    miopenTensorDescriptor_t y_acc_desc_;
    bool need_reorder_;

    bool ip_using_scratchpad() const override { return (use_acc_dst_ > 0); }
    virtual bool need_to_transform_filter() const override {
        return need_reorder_;
    }

    virtual status_t init(engine_t *, inner_product_pd_t *pd, bool with_relu,
            bool with_eltwise, bool with_sum, bool need_reorder) override {
        need_reorder_ = need_reorder;
        // GEMM is column major(so do rocblas), here the data is row major.
        // By switching the weight and source we convert the row major to
        // column major without transposing matrices.
        // B * A = C, where B is weight, A is src and C is dst
        bool wie_tr = (pd->weights_md()->format_desc.blocking.strides[0] != 1);
        CHECK(convert_data_type(pd->src_md(), &data_types_[io::src]));
        CHECK(convert_data_type(pd->weights_md(0), &data_types_[io::wei]));
        if (need_reorder) {
            CHECK(verify_format(pd->src_md()));
            ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();
            get_4d_tensor_descriptor(pd->weights_md(0), dims_[io::wei], strides_[io::wei]);
            set_filter_format(ndims_, dims_[io::wei], strides_[NUM_IO]);
            CHECK(init_filter_transformation(data_types_[io::wei], ndims_,
                    dims_[io::wei], strides_[io::wei], strides_[NUM_IO]));

            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_none,
                    memory_desc_wrapper(pd->weights_md(0)).size(), size_t(1));
            wie_tr = strides_[NUM_IO][0] != 1;
        }

        trans_a_ = wie_tr ? rocblas_operation_transpose : rocblas_operation_none;
        trans_b_ = rocblas_operation_none;
        int ic = pd->IC_total_padded();
        int oc = pd->OC();
        int mb = pd->MB();
        n_ = mb;
        k_ = ic;
        m_ = oc;
        lda_ = wie_tr ? k_ : m_;
        ldb_ = k_;
        ldc_ = m_;
        with_bias_ = pd->with_bias();
        with_eltwise_ = with_eltwise || with_relu;
        with_relu_ = with_eltwise;
        use_acc_dst_ = ((pd->dst_md()->data_type == data_type::s8)
                || (with_bias_ && pd->weights_md(1)->data_type != pd->dst_md()->data_type));
        // this must be applied on bias if exists.
        output_scales_ = pd->attr()->output_scales_.scales_[0]; // alpha
        with_sum_ = with_sum;
        // scaling factor to add the previous destination value to the current
        // computation. This is equivalent of
        sum_scale_ = sum_scale(pd);
        ndims_ = 4;

        bool input_is_blocked
                = pd->src_md()->format_desc.blocking.inner_blks[0] == 4
                && pd->weights_md(0)->format_desc.blocking.inner_blks[0] == 4;
        if (input_is_blocked) { // since we flatten the tensor and use gemm
            // we dont care about the blocked data type
            data_types_[io::src] = miopenInt8;
            data_types_[io::wei] = miopenInt8;
            data_types_[io::dst] = miopenInt8;
        } else {
            CHECK(convert_data_type(pd->dst_md(), &data_types_[io::dst]));
        }
        CHECK(get_rocblas_data_type(data_types_[io::wei], a_type_));
        CHECK(get_rocblas_data_type(data_types_[io::src], b_type_));
        c_type_ = (data_types_[io::dst] == miopenHalf && !use_acc_dst_) ? rocblas_datatype_f16_r : rocblas_datatype_f32_r;
        CHECK(set_computetype());
        
        get_4d_tensor_descriptor(pd->dst_md(), dims_[io::dst], strides_[io::dst]);
        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::dst],
                data_types_[io::dst], ndims_, dims_[io::dst],
                strides_[io::dst]));

        if (with_bias_) {
            CHECK(convert_data_type(pd->weights_md(1), &data_types_[io::bia]));
            // format is always nchw
            set_bias_dims(ndims_, pd->OC());

            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::bia],
                    data_types_[io::bia], ndims_, dims_[io::bia],
                    strides_[io::bia]));
        }
        if (use_acc_dst_) {
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt,
                    memory_desc_wrapper(pd->dst_md()).size(), size_t(1));
            CHECK(create_and_set_tensor_descriptor(&y_acc_desc_,
                    miopenFloat, ndims_, dims_[io::dst],
                    strides_[io::dst]));
        } else {
            y_acc_desc_ = tensor_descs_[io::dst];
        }
        if (with_eltwise_) { CHECK(create_and_set_op_descriptor(pd)); }
        return status::success;
    }

    // void just_check(const void* d_A, int tensor_size) const
    // {
    //     std::vector<float> host_buffer(tensor_size);
	//     hipMemcpy(host_buffer.data(), d_A, sizeof(float)*tensor_size, hipMemcpyDeviceToHost);
	//     hipDeviceSynchronize();
    //     int threshold = tensor_size - 5;
	//     printf("threshold %d +5 ele:%f, %f, %f, %f, %f\n", threshold, 
	// 	    host_buffer[threshold], host_buffer[threshold+1], host_buffer[threshold+2], host_buffer[threshold+3], host_buffer[threshold+4]);
    // }

    void execute(miopenHandle_t miopen_handle, rocblas_handle rocblas_handle_,
            const std::vector<void *> &args) const override {
        assert(args.size() == 7);
        auto x = args[0], w = args[1], b = args[2], y = args[3],
             workspace = args[4];
        auto w_arg = w;
        if (need_reorder_) {
            void *transformed_w = args[5];
            transform_filter(miopen_handle, w, transformed_w);
            w_arg = transformed_w;
        }
        auto y_dst = use_acc_dst_ ? workspace : y;
        auto sum_scale = use_acc_dst_ ? 0.0f : sum_scale_;
        // do gemm
        ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle_, trans_a_, trans_b_, 
                m_, n_, k_, &output_scales_, 
                (float*)w_arg, a_type_, lda_, 
                (float*)x, b_type_, ldb_, &sum_scale, 
                (float*)y_dst, c_type_, ldc_, 
                (float*)y_dst, c_type_, ldc_, 
                compute_type_, algo_, solution_index, flags_);

        if (with_bias_) {
            MIOPEN_EXECUTE_FUNC_V(miopenConvolutionForwardBias, miopen_handle, &bias_alpha_,
                tensor_descs_[io::bia], b, &bias_beta_, y_acc_desc_, y_dst);
        }
        
        if (use_acc_dst_) {
            // move y in workspace to user specified y
            MIOPEN_EXECUTE_FUNC(miopenTransformTensor, miopen_handle, &alpha_,
                y_acc_desc_, y_dst, &sum_scale_, tensor_descs_[io::dst], y);
        }
        if (with_eltwise_) {
            MIOPEN_EXECUTE_FUNC(miopenActivationForward, miopen_handle, act_desc_,
                    &alpha_, tensor_descs_[io::dst], y, &beta_,
                    tensor_descs_[io::dst], y);
        }
    }
    status_t create_and_set_op_descriptor(const inner_product_pd_t *pd) {
        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenCreateActivationDescriptor, &act_desc_));

        miopenActivationMode_t act_mode;
        // TODO: add more
        switch (eltwise_algorithm_kind(pd)) {
            case alg_kind::eltwise_tanh:
                act_mode = miopenActivationTANH;
                break;
            case alg_kind::eltwise_elu: 
                act_mode = miopenActivationELU; 
                break;
            case alg_kind::eltwise_relu:
                act_mode = miopenActivationRELU;
                break;
            case alg_kind::eltwise_logistic:
                act_mode = miopenActivationLOGISTIC;
                break;
            case alg_kind::eltwise_bounded_relu:
                act_mode = miopenActivationCLIPPEDRELU;
                break;
            default: return status::unimplemented;
        }
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetActivationDescriptor, act_desc_,
                act_mode, eltwise_alpha(pd), 1.0, 1.0));
        return status::success;
    }
};

struct miopen_gemm_inner_product_bwd_data_impl_t
    : public miopen_inner_product_impl_base_t,
      public miopen_gemm_inner_product_impl_base_t,
      public miopen_conv_filter_adjustment_base_t {
    bool need_reorder_;

    virtual bool need_to_transform_filter() const override {
        return need_reorder_;
    }

    // why?
    virtual status_t init(engine_t *, inner_product_pd_t *pd,
            bool /*with_relu*/, bool /*with_eltwise*/, bool /*with_sum */,
            bool need_reorder) override {
        need_reorder_ = need_reorder;

        // GEMM is column major, here the data is row major.
        // By switching the weight and source we convert the row major to
        // column major without transposing matrices.
        // B * A = C, where B is weight, A is d_dst and C is d_src
        bool wie_tr = (pd->weights_md(0)->format_desc.blocking.strides[0] == 1);
        CHECK(convert_data_type(pd->diff_src_md(), &data_types_[io::src]));
        CHECK(convert_data_type(pd->weights_md(0), &data_types_[io::wei]));
        CHECK(convert_data_type(pd->diff_dst_md(), &data_types_[io::dst]));
        if (need_reorder) {
            CHECK(verify_format(pd->diff_src_md()));
            ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();
            get_4d_tensor_descriptor(
                    pd->weights_md(0), dims_[io::wei], strides_[io::wei]);
            set_filter_format(ndims_, dims_[io::wei], strides_[NUM_IO]);
            CHECK(init_filter_transformation(data_types_[io::wei], ndims_,
                    dims_[io::wei], strides_[io::wei], strides_[NUM_IO]));

            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_none,
                    memory_desc_wrapper(pd->weights_md(0)).size(), size_t(1));
            wie_tr = strides_[NUM_IO][0] == 1;
        }
        trans_a_ = wie_tr ? rocblas_operation_transpose : rocblas_operation_none;
        trans_b_ = rocblas_operation_none;
        int ic = pd->IC_total_padded();
        int oc = pd->OC();
        int mb = pd->MB();
        n_ = mb;
        k_ = oc;
        m_ = ic;
        lda_ = wie_tr ? k_ : m_;
        ldb_ = k_;
        ldc_ = m_;
        CHECK(get_rocblas_data_type(data_types_[io::wei], a_type_));
        CHECK(get_rocblas_data_type(data_types_[io::dst], b_type_));
        CHECK(get_rocblas_data_type(data_types_[io::src], c_type_));
        return status::success;
    }
    void execute(miopenHandle_t miopen_handle, rocblas_handle rocblas_handle_,
            const std::vector<void *> &args) const override {
        assert(args.size() == 5);
        auto dx = args[0], w = args[1], dy = args[2];
        auto w_arg = w;
        if (need_reorder_) {
            void *transformed_w = args[4];
            transform_filter(miopen_handle, w, transformed_w);
            w_arg = transformed_w;
        }
        // do gemm
        ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle_, 
            trans_a_, trans_b_, m_, n_, k_, &alpha_, 
            w_arg, a_type_, lda_, 
            dy, b_type_, ldb_, &beta_, 
            dx, c_type_, ldc_,
            dx, c_type_, ldc_,
            compute_type_, algo_, solution_index, flags_);
    }
};

struct miopen_gemm_inner_product_bwd_weights_impl_t
    : public miopen_inner_product_impl_base_t,
      public miopen_gemm_inner_product_impl_base_t,
      public miopen_conv_filter_adjustment_base_t {
    miopenReduceTensorDescriptor_t reduceTensorDesc_ = nullptr;
    bool wie_tr_;
    bool need_reorder_;

    virtual bool need_to_transform_filter() const override {
        return need_reorder_;
    }

    virtual ~miopen_gemm_inner_product_bwd_weights_impl_t() {
        if (reduceTensorDesc_) {
            MIOPEN_EXECUTE_FUNC_V(
                    miopenDestroyReduceTensorDescriptor, reduceTensorDesc_);
        }
    }
    status_t create_and_set_reduce_descriptor() {
        MIOPEN_EXECUTE_FUNC_S(
                miopenCreateReduceTensorDescriptor, &reduceTensorDesc_);
        MIOPEN_EXECUTE_FUNC_S(miopenSetReduceTensorDescriptor, reduceTensorDesc_,
                MIOPEN_REDUCE_TENSOR_ADD, miopenFloat, MIOPEN_PROPAGATE_NAN,
                MIOPEN_REDUCE_TENSOR_NO_INDICES, MIOPEN_32BIT_INDICES);
        return status::success;
    }
    virtual status_t init(engine_t *engine, inner_product_pd_t *pd,
            bool /*with_relu*/, bool /*with_eltwise*/, bool /*with_sum */,
            bool need_reorder) override {
        need_reorder_ = need_reorder;
        with_bias_ = pd->with_bias();

        // GEMM is column major, here the data is row major.
        // By switching the weight and source we convert the row major to
        // column major without transposing matrices.
        // B * A = C.
        // Here backward weight is equivalent of d_dst * src^T when the weight
        // filter is IC*OC. Therefore B is d_dst and A is transposed src, and C
        // is d_wei. However, when the filter format is OC*IC , the backward
        // weight is equivalent to src * d_dst^T. In this case, B is src, A is
        // transposed d_dst and C is d_wei.
        wie_tr_ = (pd->diff_weights_md(0)->format_desc.blocking.strides[0]
                == 1);
        // std::cout << wie_tr_ << std::endl;
        CHECK(convert_data_type(pd->src_md(), &data_types_[io::src]));
        CHECK(convert_data_type(pd->diff_weights_md(0), &data_types_[io::wei]));
        CHECK(convert_data_type(pd->diff_dst_md(), &data_types_[io::dst]));
        if (need_reorder_) {
            CHECK(verify_format(pd->src_md()));
            ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();
            get_4d_tensor_descriptor(
                    pd->diff_weights_md(0), dims_[io::wei], strides_[io::wei]);
            set_filter_format(ndims_, dims_[io::wei], strides_[NUM_IO]);
            CHECK(init_filter_transformation(data_types_[io::wei], ndims_,
                    dims_[io::wei], strides_[NUM_IO], strides_[io::wei]));
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_none,
                    memory_desc_wrapper(pd->diff_weights_md(0)).size(),
                    size_t(1));
            wie_tr_ = (strides_[NUM_IO][0] == 1);
        }
        trans_a_ = rocblas_operation_none;
        trans_b_ = rocblas_operation_none;
        int ic = pd->IC_total_padded();
        int oc = pd->OC();
        int mb = pd->MB();
        n_ = wie_tr_ ? ic : oc;
        k_ = mb;
        m_ = wie_tr_ ? oc : ic;
        lda_ = m_;
        ldb_ = n_;
        ldc_ = m_;

        CHECK(get_rocblas_data_type(
                data_types_[(wie_tr_ ? io::dst : io::src)], a_type_));
        CHECK(get_rocblas_data_type(
                data_types_[(wie_tr_ ? io::src : io::dst)], b_type_));
        CHECK(get_rocblas_data_type(data_types_[io::wei], c_type_));
        if (with_bias_) {
            ndims_ = 4;
            get_4d_tensor_descriptor(pd->diff_dst_md(), dims_[io::dst], strides_[io::dst]);
            CHECK(convert_data_type(pd->diff_dst_md(), &data_types_[io::dst]));
            set_bias_dims(ndims_, pd->OC());
            CHECK(convert_data_type(
                    pd->diff_weights_md(1), &data_types_[io::bia]));
            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::dst],
                    data_types_[io::dst], ndims_, dims_[io::dst],
                    strides_[io::dst]));
            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::bia],
                    data_types_[io::bia], ndims_, dims_[io::bia],
                    strides_[io::bia]));
            CHECK(create_and_set_reduce_descriptor());

            auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
            stream_t *service_stream;
            CHECK(sycl_engine.get_service_stream(service_stream));

            auto hip_stream = utils::downcast<sycl_hip_stream_t *>(service_stream);
            auto handle = hip_stream->get_miopen_handle();

            // get the required workspace size
            MIOPEN_EXECUTE_FUNC_S(miopenGetReductionWorkspaceSize, handle,
                    reduceTensorDesc_, tensor_descs_[io::dst],
                    tensor_descs_[io::bia], &workspace_size_);
        }

        if (workspace_size_ > 0) {
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt,
                    workspace_size_, size_t(1));
        }

        return status::success;
    }
    void execute(miopenHandle_t miopen_handle, rocblas_handle rocblas_handle_,
            const std::vector<void *> &args) const override {
        assert(args.size() == 6);
        auto x = args[0], dy = args[1], dw = args[2], db = args[3],
             workspace = args[4];
        auto dw_arg = need_reorder_ ? args[5] : dw;
        // do gemm
        ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, rocblas_handle_, 
            trans_a_, trans_b_, m_, n_, k_, &alpha_, 
            (wie_tr_ ? dy : x), a_type_, lda_,
            (wie_tr_ ? x : dy), b_type_, ldb_, &beta_, 
            dw_arg, c_type_, ldc_, 
            dw_arg, c_type_, ldc_, 
            compute_type_, algo_, solution_index, flags_);

        if (need_reorder_) {
            // a user requires the oneDNN format as an output
            transform_filter(miopen_handle, dw_arg, dw);
        }
        if (with_bias_) {
            // backward bias for inner product is reduction of dy on dim[0] .
            // So we can use cudnnReduceTensor to partially reduce dy.
            MIOPEN_EXECUTE_FUNC(miopenReduceTensor, miopen_handle,
                    reduceTensorDesc_, nullptr, 0, workspace, workspace_size_,
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
