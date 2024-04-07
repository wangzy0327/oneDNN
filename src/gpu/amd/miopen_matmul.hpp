#ifndef MIOPEN_MATMUL_HPP
#define MIOPEN_MATMUL_HPP

#include <CL/sycl.hpp>

#include "common/primitive.hpp"
#include "common/matmul_pd.hpp"

#include "gpu/amd/sycl_hip_utils.hpp"
#include "common/type_helpers.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"

#include<stdio.h>

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_matmul_t : public primitive_t{
    using primitive_t::primitive_t;

    struct pd_t : public matmul_pd_t{
        using matmul_pd_t::matmul_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_matmul_t);  

        status_t init(engine_t *) {
            // TODO
            return status::success;
        }
    };

    status_t init(engine_t *engine) override {
        matmul_pd_t* pd = (matmul_pd_t *)primitive_t::pd().get();
        
        // set rocblas intput datatype
        // when A&B's datatype is i8, C&D's datatype must be f32, equals-to computetype
        CHECK(set_rocblas_data_type(pd->src_md()->data_type, Atype));
        CHECK(set_rocblas_data_type(pd->weights_md(0)->data_type, Btype));
        CHECK(set_rocblas_data_type(pd->dst_md()->data_type, Ctype));

        if(Atype == rocblas_datatype_i8_r) 
            rocblas_datatype compute_type = rocblas_datatype::rocblas_datatype_i32_r;

        // set rocblas input dimensions
        memory_desc_wrapper src_d = memory_desc_wrapper(pd->src_md());
        memory_desc_wrapper weights_d = memory_desc_wrapper(pd->weights_md(0));
        memory_desc_wrapper dst_d = memory_desc_wrapper(pd->dst_md());

        if(pd->with_bias()) return status::unimplemented;
        is_batched_ = pd->batched();
        // TODO deal eltwise
        // TODO deal runtimeparams

        // if this is a sum mm, set beta_value non-zero
        if(pd->attr()->post_ops_.contain(primitive_kind::sum, 0)
                || pd->attr()->post_ops_.contain(primitive_kind::sum, 1)){
                    int sum_idx_ = pd->attr()->post_ops_.find(primitive_kind::sum);
                    beta_value = pd->attr()->post_ops_.entry_[sum_idx_].sum.scale;  // equals 1.0f by default
                    // accumulate A*B on C as D
                    printf("set beta_value:%f\n", beta_value);
                }

        if (is_batched_) { batch_count_ = dst_d.dims()[0]; }
        M = (rocblas_int)dst_d.dims()[is_batched_ + 1];
        N = (rocblas_int)dst_d.dims()[is_batched_ + 0];
        K = (rocblas_int)src_d.dims()[is_batched_ + 1];
        
        const auto &src_strides = &src_d.blocking_desc().strides[is_batched_];
        const auto &weights_strides = &weights_d.blocking_desc().strides[is_batched_];

        // A matrix is the weights
        transA_ = (weights_strides[1] == 1 && weights_d.dims()[is_batched_ + 0] > 1) 
                    ? rocblas_operation::rocblas_operation_none 
                    : rocblas_operation::rocblas_operation_transpose;
        // B matrix is the src
        transB_ = (src_strides[1] == 1 && src_d.dims()[is_batched_ + 0] > 1)
                    ? rocblas_operation::rocblas_operation_none 
                    : rocblas_operation::rocblas_operation_transpose;

        lda_ = (int)weights_strides[!transA_];
        ldb_ = (int)src_strides[!transB_];

        const auto &dst_bd = dst_d.blocking_desc();
        ldc_ = (int)dst_bd.strides[is_batched_ + 0];

        if (is_batched_) {
            stride_a_ = (transA_ == rocblas_operation::rocblas_operation_none) ? lda_ * K : lda_ * M;
            stride_b_ = (transB_ == rocblas_operation::rocblas_operation_none) ? ldb_ * N : ldb_ * K;
            stride_c_ = ldc_ * N;
        }

        return status::success;
    }
    
    status_t execute(const exec_ctx_t &ctx) const override{
        amd::sycl_hip_stream_t *hip_stream
            = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([&](::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto wt_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);

            rocblas_handle blas_handle = hip_stream->get_rocblas_handle();

            // cgh.interop_task([=](const cl::sycl::interop_handler &ih) {
            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(hip_stream->engine());
                auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
            
                auto A = sc.memory<float *>(ih, wt_acc);
                auto B = sc.memory<float *>(ih, src_acc);
                auto C = sc.memory<float *>(ih, dst_acc);
                
                // printf("just say hello\n");
                ROCBLAS_EXECUTE_FUNC(rocblas_sgemm_strided_batched, blas_handle, transA_, transB_, M, N, K,
                    &alpha_value, A, lda_, stride_a_, B, ldb_, stride_b_, &beta_value, C, ldc_, stride_c_, batch_count_);
                
                // ROCBLAS_EXECUTE_FUNC(rocblas_sgemm_strided_batched_ex, blas_handle, transA_, transB_, M, N, K,
                //     &alpha_value, A, Atype, lda_, stride_a_, B, Btype, ldb_, stride_b_, &beta_value, C, Ctype, ldc_, stride_c_, 
                //     C, Ctype, ldc_, stride_c_, batch_count_, compute_type, algo, solution_index, flags);
            }); 
        });
    };

private:
    // gemm type    
    bool with_bias_ = false;
    bool is_batched_ = false;
    // gemm parameters

    rocblas_operation transA_ = rocblas_operation::rocblas_operation_none;     //enum 0-none 1-trans 2-c_trans
    rocblas_operation transB_ = rocblas_operation::rocblas_operation_none;
    rocblas_int M, K, N;

    float alpha_value = 1.0f;       // aka output_scale
    float beta_value = 0;

    rocblas_datatype Atype, Btype, Ctype;
    rocblas_int lda_, ldb_, ldc_;
    rocblas_stride stride_a_, stride_b_, stride_c_;
    rocblas_int batch_count_ = 1;

    // like the nvidia expand, i only use the common rocblas_datatype_f32_r compute mode here
    rocblas_datatype compute_type = rocblas_datatype::rocblas_datatype_f32_r;
    rocblas_gemm_algo algo = rocblas_gemm_algo::rocblas_gemm_algo_standard;
    int solution_index = 0;             // unused
    unsigned int flags = 0;             // optional gemm flags
    
    // set gemm parameter funcs
    status_t set_rocblas_data_type(dnnl_data_type_t dt, rocblas_datatype &blas_dt){
        // onednn support limited data type(maybe because of sycl?)
        switch (dt)
        {
            case dnnl_data_type_t::dnnl_f16:
                blas_dt = rocblas_datatype_f16_r;   // float 16 r means real
                return status::success;
            case dnnl_data_type_t::dnnl_bf16:   
                blas_dt = rocblas_datatype_bf16_r;  // bfloat 16
                return status::success;
            case dnnl_data_type_t::dnnl_f32:
                blas_dt = rocblas_datatype_f32_r;   // float 32
                return status::success;
            case dnnl_data_type_t::dnnl_s8:
                blas_dt = rocblas_datatype_i8_r;    // int 8
                return status::success;
        }
        return status::unimplemented;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif