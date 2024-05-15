#ifndef CNNL_MATMUL_HPP
#define CNNL_MATMUL_HPP

#include <CL/sycl.hpp>

#include "common/primitive.hpp"
#include "common/matmul_pd.hpp"

#include "gpu/cambricon/sycl_bang_utils.hpp"
#include "common/type_helpers.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_scoped_context.hpp"
#include "gpu/cambricon/sycl_bang_stream.hpp"

#include<stdio.h>

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_matmul_t : public primitive_t{
    using primitive_t::primitive_t;

    struct pd_t : public matmul_pd_t{
        using matmul_pd_t::matmul_pd_t;

        DECLARE_COMMON_PD_T("bang:cnnl:any", cnnl_matmul_t);

        status_t init(engine_t *) {
            // TODO
            return status::success;
        }
    };

    status_t init(engine_t *engine) override {
        matmul_pd_t* pd = (matmul_pd_t *)primitive_t::pd().get();
        
        // set cnnl intput datatype
        // when A&B's datatype is i8, C&D's datatype must be f32, equals-to computetype
        CHECK(set_cnnl_data_type(pd->src_md()->data_type, Atype));
        CHECK(set_cnnl_data_type(pd->weights_md(0)->data_type, Btype));
        CHECK(set_cnnl_data_type(pd->dst_md()->data_type, Ctype));

        // set cnnl input dimensions
        memory_desc_wrapper src_d = memory_desc_wrapper(pd->src_md());
        memory_desc_wrapper weights_d = memory_desc_wrapper(pd->weights_md(0));
        memory_desc_wrapper dst_d = memory_desc_wrapper(pd->dst_md());
        
        is_batched_ = pd->batched();

        // TODO: deal eltwise
        // TODO: deal runtimeparams

        // if this is a sum mm, set mm_beta non-zero
        if(pd->attr()->post_ops_.contain(primitive_kind::sum, 0)
                || pd->attr()->post_ops_.contain(primitive_kind::sum, 1)){
                    int sum_idx_ = pd->attr()->post_ops_.find(primitive_kind::sum);
                    mm_beta = pd->attr()->post_ops_.entry_[sum_idx_].sum.scale;  // equals 1.0f by default
                    // accumulate A*B on C as D
                    printf("set mm_beta:%f\n", mm_beta);
                }

        if (is_batched_) { batch_count_ = dst_d.dims()[0]; }
        
        convert_dims(src_d.dims(), A_dims, is_batched_ + 2);
        convert_dims(weights_d.dims(), B_dims, is_batched_ + 2);
        convert_dims(dst_d.dims(), C_dims, is_batched_ + 2);

        // M = (int)dst_d.dims()[is_batched_ + 0];
        // N = (int)dst_d.dims()[is_batched_ + 1];
        // K = (int)src_d.dims()[is_batched_ + 1];
        // A_dims[0] = M; A_dims[1] = K;
        // B_dims[0] = K; B_dims[1] = N;
        // C_dims[0] = M; C_dims[1] = N;

        with_bias_ = pd->with_bias();
        std::cout<<"init with_bias_ flag is : "<<with_bias_<<std::endl;
        if(with_bias_)
        {   
            Bias_dims[0] = C_dims[is_batched_ + 1];
            // Bias_dims[1] = N;
        }

        CHECK(create_cnnl_descs());

        const auto &src_strides = &src_d.blocking_desc().strides[is_batched_];
        const auto &weights_strides = &weights_d.blocking_desc().strides[is_batched_];

        // A matrix is the weights
        if(src_d.dims()[0+is_batched_] == dst_d.dims()[0+is_batched_])
            transA_ = false;
        else if(src_d.dims()[1+is_batched_] == dst_d.dims()[0+is_batched_])
            transA_ = true;
        else
            return status::invalid_arguments;

        if(!transA_)
        {
            if(src_d.dims()[1+is_batched_] == weights_d.dims()[0+is_batched_]) 
                transB_ = false;
            else if(src_d.dims()[1+is_batched_] == weights_d.dims()[1+is_batched_])
                transB_ = true;
            else
                return status::invalid_arguments;
        }
        else
        {
            if(src_d.dims()[0+is_batched_] == weights_d.dims()[0+is_batched_]) 
                transB_ = false;
            else if(src_d.dims()[0+is_batched_] == weights_d.dims()[1+is_batched_])
                transB_ = true;
            else
                return status::invalid_arguments;
        }

        // init scratchpad
        init_scratchpad(engine);

        return status::success;
    }

    status_t init_scratchpad(engine_t *engine) {
        std::cout<<"init scratchpad"<<std::endl;
        auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto bang_stream = utils::downcast<sycl_bang_stream_t *>(service_stream);
        auto handle = bang_stream->get_cnnl_handle();
        
        scratchpad_size = 0;
        // CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetBiasAddWorkspaceSize, handle,
        //         Bias_desc, C_desc, &scratchpad_size_biasadd));
        std::cout<<"init biasAdd scratchpad size is "<<scratchpad_size_biasadd<<std::endl;        
        scratchpad_size = std::max(scratchpad_size_biasadd, scratchpad_size);

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetQuantizeParamWorkspaceSize, handle, A_desc, &scratchpad_size_qA));
        std::cout<<"init cnnlGetQuantizeParam scratchpad_size_qA size is "<<scratchpad_size_qA<<std::endl;        
        scratchpad_size = std::max(scratchpad_size_qA, scratchpad_size);
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetQuantizeParamWorkspaceSize, handle, B_desc, &scratchpad_size_qB));
        std::cout<<"init cnnlGetQuantizeParam scratchpad_size_qB size is "<<scratchpad_size_qB<<std::endl;        
        scratchpad_size = std::max(scratchpad_size_qB, scratchpad_size);

        // TODO: compare with the way to allocate scratchpad in other kernels.
        init_scratch_buffer();

        return status::success;
    }

    status_t create_cnnl_descs() {
        // Here, format[x] and format[y] should be NHWC
        CHECK(create_and_set_tensor_descriptor(&A_desc, CNNL_LAYOUT_ARRAY, Atype, is_batched_+2, A_dims));
        CHECK(create_and_set_tensor_descriptor(&quantized_A_desc, CNNL_LAYOUT_ARRAY, quantized_dtype, is_batched_+2, A_dims));
        
        CHECK(create_and_set_tensor_descriptor(&B_desc, CNNL_LAYOUT_ARRAY, Btype, is_batched_+2, B_dims));  
        CHECK(create_and_set_tensor_descriptor(&quantized_B_desc, CNNL_LAYOUT_ARRAY, quantized_dtype, is_batched_+2, B_dims));
        
        CHECK(create_and_set_tensor_descriptor(&C_desc, CNNL_LAYOUT_ARRAY, Ctype, is_batched_+2, C_dims));

        if (with_bias_) {
            CHECK(create_and_set_tensor_descriptor(&Bias_desc, CNNL_LAYOUT_ARRAY, Ctype, 1, Bias_dims));
        }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override{
        cambricon::sycl_bang_stream_t *bang_stream
            = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());
        
        return bang_stream->interop_task([&](::sycl::handler &cgh) {
            // auto arg = &(ctx.input(DNNL_ARG_SRC) ? *(ctx.input(DNNL_ARG_SRC)->memory_storage()) 
            //         : dnnl::impl::memory_storage_t::empty_storage());
            // auto src_acc = utils::downcast<sycl::sycl_buffer_memory_storage_t *>(arg)->buffer().get_access<cl::sycl::access::mode::read>(cgh);
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto wt_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            std::shared_ptr<
                    ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::read>>
                    bias_acc;
            if(with_bias_)
            {
                std::cout<<"with bias"<<std::endl;
                bias_acc = std::make_shared<
                        ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::read>>(
                        CTX_IN_ACCESSOR(DNNL_ARG_BIAS));
            }

            using scratch_acc_t = ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::read_write>;
            std::shared_ptr<scratch_acc_t> p_scratch_acc;

            if (scratchpad_size > 0) {
                std::cout<<"scratchped size > 0"<<std::endl;
                p_scratch_acc = std::make_shared<scratch_acc_t>(
                        scratch_buff_
                                ->get_access<::sycl::access::mode::read_write>(
                                        cgh));
            }
            
            cnnlHandle_t handle = bang_stream->get_cnnl_handle();

            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(bang_stream->engine());
                auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            
                auto A = sc.memory<float *>(ih, src_acc);
                auto B = sc.memory<float *>(ih, wt_acc);  
                auto C = sc.memory<float *>(ih, dst_acc);
                void* bias = nullptr;
                if(with_bias_)
                    bias = sc.memory<float *>(ih, *bias_acc);

                void* scratchpad = nullptr;
                if(scratchpad_size > 0)
                    scratchpad = sc.memory<float *>(ih, *p_scratch_acc);

                // quantization
                // These quantized tensors should be saved for backward, but not implemented yet
                void *d_q_B, *d_q_A;
                
                int src_size = is_batched_ ? A_dims[0]*A_dims[1]*A_dims[2] : A_dims[0]*A_dims[1];
                int weight_size = is_batched_ ? B_dims[0]*B_dims[1]*B_dims[2] : B_dims[0]*B_dims[1];
                
                cnrtMalloc(&d_q_A, sizeof(int16_t) * src_size);
                cnrtMalloc(&d_q_B, sizeof(int16_t) * weight_size);
                cnrtMemset(d_q_A, 0, sizeof(int16_t) * src_size);
                cnrtMemset(d_q_B, 0, sizeof(int16_t) * weight_size);
                
                cnrtSyncDevice();
                
                // Quantize input
                void* scratchpad_qA = scratchpad_size_qA > 0 ? scratchpad : nullptr;
                if(scratchpad_qA)
                    quantize_array(handle, A_desc, A, 16, scratchpad_qA, scratchpad_size_qA, quantized_A_desc, d_q_A);                
                // Quantize weight
                void* scratchpad_qB = scratchpad_size_qB > 0 ? scratchpad : nullptr;
                if(scratchpad_qB)
                    quantize_array(handle, B_desc, B, 16, scratchpad_qB, scratchpad_size_qB, quantized_B_desc, d_q_B);
                if(is_batched_)
                {
                    std::cout<<"matmul is batched !"<<std::endl;
                    if(scratchpad_qA && scratchpad_qB){
                        std::cout<<"matmul batched quantized !"<<std::endl;
                        CNNL_EXECUTE_FUNC(cnnlBatchMatMul, handle, transA_, transB_, 
                            quantized_A_desc, d_q_A, quantized_B_desc, d_q_B, C_desc, C);   
                    }else{
                        std::cout<<"matmul batched no quantized !"<<std::endl;
                        CNNL_EXECUTE_FUNC(cnnlBatchMatMul, handle, transA_, transB_, 
                            A_desc, d_q_A, B_desc, d_q_B, C_desc, C); 
                    }                    
                }
                else
                {
                    std::cout<<"matmul is no batched !"<<std::endl;
                    if(scratchpad_qA && scratchpad_qB){
                        std::cout<<"matmul no batched quantized !"<<std::endl;
                        CNNL_EXECUTE_FUNC(cnnlMatMul, handle, transA_, transB_, &mm_alpha,
                            quantized_A_desc, d_q_A, quantized_B_desc, d_q_B, &mm_beta, C_desc, C);
                    }else{
                        std::cout<<"matmul no batched no quantized !"<<std::endl;
                        CNNL_EXECUTE_FUNC(cnnlBatchMatMul, handle, transA_, transB_, 
                            A_desc, d_q_A, B_desc, d_q_B, C_desc, C);
                    }                   
                }
                if(with_bias_)
                {   
                    std::cout<<"matmul is with bias !"<<std::endl;
                    void* scratchpad_biasadd = scratchpad_size_biasadd > 0 ? scratchpad : nullptr;
                    CNNL_EXECUTE_FUNC(cnnlBiasAdd, handle, &bias_alpha, Bias_desc, bias, 
                            scratchpad_biasadd, scratchpad_size_biasadd, &bias_beta, C_desc, C);
                }
            });
        });
    };

private:
    // gemm type    
    bool with_bias_ = false;
    bool is_batched_ = false;
    // gemm parameters

    bool transA_ = false;
    bool transB_ = false;
    
    int A_dims[4];
    int B_dims[4];
    int C_dims[4];
    int Bias_dims[4];

    float mm_alpha = 1.0f;       // aka output_scale
    float mm_beta = 0;

    float bias_alpha = 1.0f;
    float bias_beta = 1.0f;

    cnnlTensorDescriptor_t A_desc, B_desc, C_desc, Bias_desc;
    cnnlTensorDescriptor_t quantized_A_desc, quantized_B_desc;

    cnnlDataType_t Atype, Btype, Ctype;
    cnnlDataType_t quantized_dtype = CNNL_DTYPE_INT16;

    int batch_count_ = 1;

    std::size_t scratchpad_size = 0;
    std::size_t scratchpad_size_biasadd = 0;
    std::size_t scratchpad_size_qA = 0;
    std::size_t scratchpad_size_qB = 0;

    std::shared_ptr<::sycl::buffer<uint8_t, 1>> scratch_buff_ {nullptr};    // workspace

    // set gemm parameter funcs
    status_t set_cnnl_data_type(dnnl_data_type_t dt, cnnlDataType_t &blas_dt){
        // onednn support limited data type(maybe because of sycl?)
        switch (dt)
        {
            // case dnnl_data_type_t::dnnl_f16:
            //     blas_dt = CNNL_DTYPE_HALF;   // float 16 r means real
            //     return status::success;
            case dnnl_data_type_t::dnnl_f32:
                blas_dt = CNNL_DTYPE_FLOAT;   // float 32
                return status::success;
            case dnnl_data_type_t::dnnl_s8:
                blas_dt = CNNL_DTYPE_INT8;    // int 8
                return status::success;
        }
        printf("this gemm only support float32 now\n");
        return status::unimplemented;
    }
    
    void init_scratch_buffer() {
        if (scratchpad_size > 0) {
            scratch_buff_.reset(new ::sycl::buffer<uint8_t, 1>(scratchpad_size));
        }
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif