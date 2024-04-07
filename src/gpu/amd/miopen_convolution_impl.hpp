#ifndef GPU_AMD_MIOPEN_CONVOLUTION_IMPL_HPP
#define GPU_AMD_MIOPEN_CONVOLUTION_IMPL_HPP

#include "miopen/miopen.h"

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/utils.hpp"
#include "gpu/amd/miopen_conv_filter_adjustment_base.hpp"
#include "gpu/amd/miopen_convolution_pd.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

// #include <chrono>
#include <iostream>

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_convolution_impl_base_t
    : public miopen_conv_filter_adjustment_base_t {
protected:
    enum io { x = 0, bias, weights, y, NUM_IO };
    memory_desc_t dnnl_descs[NUM_IO];
    miopenConvolutionDescriptor_t conv_desc;
    int padding[8]; //CUDNN_DIM_MAX=8, i don't know
    int dilation[8];
    miopenTensorDescriptor_t descs[NUM_IO];
    miopenDataType_t data_types[NUM_IO];
    int ndims[NUM_IO];
    int dims[NUM_IO][DNNL_MAX_NDIMS];
    int strides[NUM_IO + 1][DNNL_MAX_NDIMS];
    int filter_strides[DNNL_MAX_NDIMS];
    // cudnnTensorFormat_t formats[NUM_IO]; miopen have no tensorFormat, since it only support nchw
    bool filter_needs_transform = false;
    // cudnnFilterDescriptor_t weights_desc; no!!!
    miopenTensorDescriptor_t weights_desc;
    float alpha = 0.f;
    float beta = 0.f;
    int group_count = 1;
    bool with_groups = false;
    size_t scratchpad_size = 0;
    bool with_bias = false;

    bool do_scaling = false;
    float output_scaling = 1.0f;
    bool use_temp_dst_ = false;
    miopenDataType_t computation_data_type = miopenFloat;
    miopenDataType_t reorder_type = miopenInt8;

public:
    virtual ~miopen_convolution_impl_base_t() {
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, weights_desc);
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyConvolutionDescriptor, conv_desc);
        for (size_t i = 0; i < io::NUM_IO; i++) {
            MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, descs[i]);
        }
    }
    virtual status_t configure_alg_kind(engine_t *, convolution_pd_t *pd) = 0;

    virtual bool supported_filter_format(
            const memory_desc_t *md) const override {
        const memory_desc_wrapper mem_wrapper(md);

        return (mem_wrapper.matches_one_of_tag(format_tag::ab, format_tag::abc,
                        format_tag::abcd, format_tag::abcde, format_tag::abcdef)
                || (with_groups ? mem_wrapper.matches_one_of_tag(
                            format_tag::gowi, format_tag::gohwi,
                            format_tag::godhwi)
                                : mem_wrapper.matches_one_of_tag(
                                        format_tag::owi, format_tag::ohwi,
                                        format_tag::odhwi)));
    }

    bool using_transformed_filter() const { return filter_needs_transform; }
    bool with_scratchpad() const { return scratchpad_size > 0; }

    virtual status_t init(engine_t *engine, convolution_pd_t *pd,
            bool use_scratch_dst = false) {
        CHECK(configure_parameters(pd));
        CHECK(create_miopen_descs(pd));
        CHECK(check_output_dims());
        CHECK(configure_alg_kind(engine, pd));
        CHECK(init_scratchpad(engine, pd));

        return status::success;
    }

    virtual status_t init_zero_dims(convolution_pd_t *pd) {
        return status::success;
    }
    void get_dims_and_strides(int io) {
        convert_dims(
                dnnl_descs[io].dims, dims[io], dnnl_descs[io].ndims, ndims[io]);
        if (ndims[io] > dnnl_descs[io].ndims) {
            std::swap(dims[io][ndims[io] - 1], dims[io][ndims[io] - 2]);
            if (ndims[io] == 4) {
                propagate_strides(strides[io], dims[io], {3, 2, 1, 0}); // same reason
            }
        } else {
            convert_dims(dnnl_descs[io].format_desc.blocking.strides,
                    strides[io], dnnl_descs[io].ndims, ndims[io]);
        }
    }

    status_t configure_parameters(const convolution_pd_t *pd) {
        if (pd->ndims() > 8) { return status::invalid_arguments; }
        CHECK(set_padding_and_dilation(pd));
        with_groups = pd->with_groups();
        with_bias = pd->with_bias();
        alpha = 1.0f;
        beta = 0.0f;
        output_scaling = pd->attr()->output_scales_.scales_[0];
        do_scaling = output_scaling != 1.f;
        dnnl_descs[x] = *pd->invariant_src_md();
        dnnl_descs[weights] = *pd->invariant_wei_md();
        dnnl_descs[y] = *pd->invariant_dst_md();
        if (with_bias) dnnl_descs[bias] = *pd->invariant_bia_md();

        ndims[x] = std::max(dnnl_descs[x].ndims, 4);
        ndims[weights] = std::max(dnnl_descs[weights].ndims, 4 + with_groups);
        ndims[y] = std::max(dnnl_descs[y].ndims, 4);

        CHECK(convert_data_type(&dnnl_descs[x], &data_types[x]));
        CHECK(convert_data_type(&dnnl_descs[weights], &data_types[weights]));
        CHECK(convert_data_type(&dnnl_descs[y], &data_types[y]));

        CHECK(verify_formats());
        set_compute_format();
        get_dims_and_strides(x);
        get_dims_and_strides(weights);
        get_dims_and_strides(y);

        if (!supported_filter_format(&dnnl_descs[weights])) {
            set_filter_format(
                    ndims[weights], dims[weights], strides[NUM_IO]);
            CHECK(init_filter_transformation(data_types[weights],
                    ndims[weights], dims[weights], strides[weights],
                    strides[NUM_IO]));
            filter_needs_transform = true;
        } else {
            CHECK(get_filter_format());
            get_dims_and_strides(weights);
        }
        if (with_groups) {
            dims[weights][1] *= pd->G();
            ndims[weights] = std::max(4, ndims[weights] - with_groups);
        }

        if (with_bias) {
            ndims[bias] = dnnl_descs[bias].ndims;
            CHECK(convert_data_type(&dnnl_descs[bias], &data_types[bias]));
            convert_dims(
                    dnnl_descs[bias].dims, dims[bias], ndims[bias], ndims[y]);
            std::swap(dims[bias][0], dims[bias][1]);
            convert_dims(dnnl_descs[bias].format_desc.blocking.strides,
                    strides[bias], ndims[bias], ndims[y]);
            ndims[bias] = ndims[y];
        }

        return status::success;
    }

    status_t create_miopen_descs(const convolution_pd_t *pd) {
        CHECK(create_and_set_convolution_desc(pd));
        CHECK(create_and_set_tensor_descriptor(
                &descs[x], data_types[x], ndims[x], dims[x], strides[x]));
        CHECK(create_and_set_tensor_descriptor(&weights_desc,
                data_types[weights], ndims[weights],
                dims[weights] + with_groups, strides[weights]));
        CHECK(create_and_set_tensor_descriptor(
                &descs[y], data_types[y], ndims[y], dims[y], strides[y]));

        if (with_bias) {
            CHECK(create_and_set_tensor_descriptor(&descs[bias],
                    data_types[bias], ndims[bias], dims[bias], strides[bias]));
        }
        return status::success;
    }

    virtual status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) {
        if (filter_needs_transform) {
            auto sz = memory_desc_wrapper(&dnnl_descs[weights]).size();
            auto data_size
                    = types::data_type_size(pd->invariant_wei_md(0)->data_type);
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_cudnn_filter, sz,
                    data_size);
        }
        return status::success;
    };

    status_t create_and_set_convolution_desc(const convolution_pd_t *pd) {
        MIOPEN_EXECUTE_FUNC_V(miopenCreateConvolutionDescriptor, &conv_desc);

        MIOPEN_EXECUTE_FUNC_V(miopenInitConvolutionNdDescriptor, conv_desc,
                ndims[x] - 2, padding, filter_strides, dilation,
                miopenConvolutionMode_t::miopenConvolution);
        // Check for groups and set group count if necessary
        if (with_groups) {
            group_count = pd->G();
            if (group_count > 1)
                CHECK(MIOPEN_EXECUTE_FUNC_S(
                        miopenSetConvolutionGroupCount, conv_desc, group_count));
        }
        return status::success;
    }

    status_t set_padding_and_dilation(const convolution_pd_t *pd) {
        int actual_ndims = pd->ndims();
        if (actual_ndims == 3) {
            padding[0] = 0;
            padding[1] = static_cast<int>(pd->padL());
            dilation[0] = 1;
            dilation[1] = static_cast<int>(pd->KDW() + 1);

            filter_strides[0] = 1;
            filter_strides[1] = static_cast<int>(pd->KSW());
        } else if (actual_ndims == 4) {
            padding[0] = static_cast<int>(pd->padT());
            padding[1] = static_cast<int>(pd->padL());

            dilation[0] = static_cast<int>(pd->KDH());
            dilation[1] = static_cast<int>(pd->KDW());

            filter_strides[0] = static_cast<int>(pd->KSH());
            filter_strides[1] = static_cast<int>(pd->KSW());
        } else {
            padding[0] = static_cast<int>(pd->padFront());
            padding[1] = static_cast<int>(pd->padT());
            padding[2] = static_cast<int>(pd->padL());

            dilation[0] = static_cast<int>(pd->KDD());
            dilation[1] = static_cast<int>(pd->KDH());
            dilation[2] = static_cast<int>(pd->KDW());

            filter_strides[0] = static_cast<int>(pd->KSD());
            filter_strides[1] = static_cast<int>(pd->KSH());
            filter_strides[2] = static_cast<int>(pd->KSW());
        }
        return status::success;
    }

    virtual void execute(miopenHandle_t handle, const std::vector<void *> &args) const = 0;

    void execute_sum(miopenHandle_t handle, void *x, void *y, float alpha_,
            float beta_) const {
        float alpha1 = alpha_;
        float alpha2 = beta_;
        float beta = 0;

        MIOPEN_EXECUTE_FUNC_V(miopenOpTensor, handle, miopenTensorOp_t::miopenTensorOpAdd, &alpha1, descs[io::y], x,
                &alpha2, descs[io::y], y, &beta, descs[io::y], y);
    }

    void execute_scale(miopenHandle_t handle, void *y) const {
        if (do_scaling) {
            MIOPEN_EXECUTE_FUNC_V(
                    miopenScaleTensor, handle, descs[io::y], y, &output_scaling);
        }
    }

    void execute_set_weights_bias(
            miopenHandle_t handle, void *weights, void *bias, float value) {
        MIOPEN_EXECUTE_FUNC_V(
                miopenSetTensor, handle, descs[io::weights], weights, &value);
        if (bias) {
            MIOPEN_EXECUTE_FUNC_V(
                    miopenSetTensor, handle, descs[io::bias], bias, &value);
        }
    }

    bool with_eltwise(const convolution_pd_t *pd, int position) const {
        return pd->attr()->post_ops_.contain(primitive_kind::eltwise, position);
    }

    status_t check_output_dims() const {
        int expected_dims[8] = {};
        int nd[8] = {};
        // why? what will nd be?
        MIOPEN_EXECUTE_FUNC_V(miopenGetConvolutionNdForwardOutputDim, conv_desc,
                descs[x], weights_desc, &nd[0], &expected_dims[0]);
        for (size_t i = 0; i < ndims[y]; i++) {
            if (dims[y][i] != expected_dims[i]) return status::unimplemented;
        }
        return status::success;
    }

    void set_compute_format() {
        if (data_types[x] == miopenInt8) {
            computation_data_type = miopenInt32;
        } else {
            computation_data_type = data_types[y];
        }
    }

    status_t get_filter_format() {
        memory_desc_wrapper wrapper(&dnnl_descs[weights]);
        if (wrapper.matches_one_of_tag(format_tag::ab, format_tag::abc,
                    format_tag::abcd, format_tag::abcde, format_tag::abcdef))
            return status::success;
        else 
            return status::unimplemented;
    }

    status_t verify_formats() {
        CHECK(verify_format(&dnnl_descs[x]));
        CHECK(verify_format(&dnnl_descs[y]));
        return status::success;
    }

    bool use_temp_dst() const { return use_temp_dst_; }
};

struct miopen_convolution_impl_fwd_t : public miopen_convolution_impl_base_t {
protected:
    miopenActivationDescriptor_t activation_desc = nullptr;
    miopenActivationDescriptor_t eltwise_desc = nullptr;
    miopenTensorDescriptor_t reorder_dst_desc = nullptr;
    miopenConvSolution_t fwd_solution;
    std::vector<miopenConvSolution_t> perf;
    size_t requested_algo_count = 0;
    size_t returned_algo_count = 0;
    int num_post_ops = 0;
    primitive_kind_t post_ops[2];
    bool need_reorder = false;
    float sum_scale = 1.0f;
    bool conv_bias_eltwise = false;
    bool conv_bias = false;

public:
    // destructor
    virtual ~miopen_convolution_impl_fwd_t() {
        if (activation_desc)
            MIOPEN_EXECUTE_FUNC_V(
                    miopenDestroyActivationDescriptor, activation_desc);
        if (eltwise_desc)
            MIOPEN_EXECUTE_FUNC_V(
                    miopenDestroyActivationDescriptor, eltwise_desc);
        if (reorder_dst_desc)
            MIOPEN_EXECUTE_FUNC_V(
                    miopenDestroyTensorDescriptor, reorder_dst_desc);
    }

    // fuse eltwise, bias and convolution
    status_t configure_post_ops(convolution_pd_t *pd) {
        auto &p = pd->attr()->post_ops_;
        num_post_ops = p.len();
        for (size_t i = 0; i < p.len(); i++) {
            post_ops[i] = p.entry_[i].kind;
            if (post_ops[i] == dnnl_eltwise) {
                CHECK(create_and_set_eltwise_descriptor(pd));
            }
            if (post_ops[i] == dnnl_sum) { sum_scale = p.entry_[i].sum.scale; }
        }

        // Try to fuse kernels
        // pattern 1: conv + bias + eltwise
        conv_bias_eltwise = num_post_ops > 0 && post_ops[0] == dnnl_eltwise
                && with_bias && !do_scaling
                && data_types[y] != miopenInt8
                // XXX: cuDNN has a correctness issue for fusion of group conv
                && pd->G() == 1
                && eltwise_algorithm_kind(pd) == alg_kind::eltwise_relu;
        // pattern 2: conv + bias
        conv_bias = with_bias && !conv_bias_eltwise
                && !do_scaling
                // XXX: cuDNN limitation on algorithm support when activation is
                // equal to CUDNN_ACTIVATION_IDENTITY.
                && fwd_solution.algorithm
                        == miopenConvolutionFwdAlgoImplicitGEMM
                // XXX: cuDNN has a correctness issue for fusion of group conv
                && pd->G() == 1;
        // If the only post-op is fused then there is no need for temp dst
        if (conv_bias_eltwise && num_post_ops == 1) use_temp_dst_ = false;

        if (data_types[y] == miopenInt8 && use_temp_dst_) {
            data_types[y] = miopenFloat;
            need_reorder = true;
            CHECK(create_and_set_tensor_descriptor(&reorder_dst_desc,
                reorder_type, ndims[y], dims[y], strides[y]));
        }

        return status::success;
    }

    status_t init(engine_t *engine, convolution_pd_t *pd,
            bool use_scratch_dst) override {
        use_temp_dst_ = use_scratch_dst;
        CHECK(configure_parameters(pd));
        CHECK(create_miopen_descs(pd));
        CHECK(configure_alg_kind(engine, pd));
        CHECK(configure_post_ops(pd));
        CHECK(init_scratchpad(engine, pd));

        return status::success;
    }

    void execute_reorder(miopenHandle_t handle, void *src, void *dst, bool flip_formats) const {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        if (flip_formats) {
            MIOPEN_EXECUTE_FUNC_V(miopenTransformTensor, handle, &alpha,
                    reorder_dst_desc, src, &beta, descs[y], dst);
        } else {
            MIOPEN_EXECUTE_FUNC_V(miopenTransformTensor, handle, &alpha, descs[y],
                    src, &beta, reorder_dst_desc, dst);
        }
    }

    void execute_eltwise(miopenHandle_t handle, void *src, void *dst) const {
        float alpha = 1.0f;
        float beta = 0.0f;
        MIOPEN_EXECUTE_FUNC_V(miopenActivationForward, handle, eltwise_desc,
                &alpha, descs[io::y], src, &beta, descs[io::y], dst);
    }

    void execute(miopenHandle_t handle, const std::vector<void *> &args) const override {
        auto x = args[0], weights = args[1], y = args[2], bias = args[3],
             scratchpad = args[4], post_op_scratch = args[6],
             post_op_reorder = args[7];
        void *output = use_temp_dst_ ? post_op_scratch : y;
        if (using_transformed_filter()) {
            auto w_scratch = args[5];
            transform_filter(handle, weights, w_scratch);
            weights = w_scratch;
        }
        bool fused = conv_bias || conv_bias_eltwise;

        if(fused){
            // TODO Fused layer 
            const float bias_alpha = 1.0f;
            const float bias_beta = 1.0f;
            const float activ_alpha = 1.0f;
            const float activ_beta = 0;
            
            MIOPEN_EXECUTE_FUNC_V(miopenConvolutionForwardImmediate, handle, 
                    weights_desc, weights, descs[io::x], x, conv_desc, descs[io::y], output, 
                    scratchpad, scratchpad_size, fwd_solution.solution_id);

            MIOPEN_EXECUTE_FUNC_V(miopenConvolutionForwardBias, handle, &bias_alpha,
                        descs[io::bias], bias, &bias_beta, descs[io::y],
                        output);
            MIOPEN_EXECUTE_FUNC_V(miopenActivationForward, handle, 
            conv_bias_eltwise ? eltwise_desc : activation_desc,
            &activ_alpha, descs[io::y], output, &activ_beta, 
            descs[io::y], output);
        }
        else{
            // std::cout << "selcet algo: " << fwd_solution.algorithm << "\n";
            // printf("ws_size:%d\n", scratchpad_size);
            // hipDeviceSynchronize();
            // auto start = std::chrono::system_clock::now();
            
            MIOPEN_EXECUTE_FUNC_V(miopenConvolutionForwardImmediate, handle, 
                weights_desc, weights, descs[io::x], x, conv_desc, descs[io::y], output, 
                scratchpad, scratchpad_size, fwd_solution.solution_id);

            // printf("conv algo:%d                                           \n", fwd_solution.solution_id);
            // hipDeviceSynchronize();
            // auto end = std::chrono::system_clock::now();
            // auto time_compute = duration_cast<std::chrono::microseconds>(end-start);          
            // std::cout << "time compute:" <<
            //     float(time_compute.count()) * std::chrono::microseconds::period::num/std::chrono::microseconds::period::den 
            //     << "\n";

            const float bias_alpha = 1.0f;
            const float bias_beta = 0.0f;           
            if (with_bias) {
                MIOPEN_EXECUTE_FUNC_V(miopenConvolutionForwardBias, handle, &bias_alpha,
                    descs[io::bias], bias, &bias_beta, descs[io::y], output);
            }
        }
        
        execute_scale(handle, output);

        // skip first eltwise in case it is fused into convolution
        const int post_ops_start_pos = fused && conv_bias_eltwise;
        for (int i = post_ops_start_pos; i < num_post_ops; i++) {
            bool last_op = i == num_post_ops - 1 && !need_reorder;
            switch (post_ops[i]) {
                case dnnl_sum:
                    if (need_reorder) {
                        execute_reorder(handle, y, post_op_reorder, true);
                        execute_sum(handle, post_op_reorder, post_op_scratch,
                                sum_scale, 1.0f);
                    } else if (last_op) {
                        execute_sum(
                                handle, post_op_scratch, y, 1.0f, sum_scale);
                    } else {
                        execute_sum(
                                handle, y, post_op_scratch, sum_scale, 1.0f);
                    }

                    break;

                case dnnl_eltwise:
                    if (last_op) {
                        execute_eltwise(handle, output, y);
                    } else {
                        execute_eltwise(handle, output, post_op_scratch);
                    }
                    break;
                default: assert(!"unsupported post op");
            }
        }

        if (need_reorder) {
            execute_reorder(handle, post_op_scratch, y, false);
        }
    }

    // workspace init
    status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto hip_stream
                = utils::downcast<sycl_hip_stream_t *>(service_stream);
        auto handle = hip_stream->get_miopen_handle();
        
        // TODO: GEMM sulotion not support
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenConvolutionForwardGetSolutionWorkspaceSize,
                handle, weights_desc, descs[x], conv_desc, descs[y], 
                fwd_solution.solution_id, &scratchpad_size));

        if (scratchpad_size > 0)
            // let try use cudnn's key
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_cudnn_algo,    
                    scratchpad_size, size_t(1));

        return miopen_convolution_impl_base_t::init_scratchpad(engine, pd);
    }

    // find algorithm kind
    status_t configure_alg_kind(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
        hip_sycl_scoped_context_handler_t sc(sycl_engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto hip_stream = utils::downcast<sycl_hip_stream_t *>(service_stream);
        auto handle = hip_stream->get_miopen_handle();

        // refer to https://github.com/ROCmSoftwarePlatform/MIOpen/blob/develop/doc/src/find_and_immediate.md
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenConvolutionForwardGetSolutionCount,
                handle, weights_desc, descs[x], conv_desc, descs[y], &requested_algo_count));

        perf.resize(requested_algo_count);
        
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenConvolutionForwardGetSolution, handle,
                weights_desc, descs[x], conv_desc, descs[y],
                requested_algo_count, &returned_algo_count, perf.data()));

        // Specify the algo, just for debug use.
        // printf("useable algo:%d\n", returned_algo_count);
        // bool algo_selected = false;
        // for(int i = 0; i<returned_algo_count; i++){
        //     printf("algo: %d, ws_size:%d\n", perf[i].algorithm, perf[i].workspace_size);
        //     if((int)perf[i].algorithm == 1){
        //         fwd_solution = perf[i];
        //         algo_selected = true;
        //     }
        // }
        // printf("\n");
        // if(!algo_selected)
        // {
        //     printf("direct conv not avilable!\n");
        //     return status::unimplemented;
        // }

        for (size_t i = 0; i < returned_algo_count; i++) {
            switch (pd->desc()->alg_kind) {
                case dnnl_convolution_auto:
                    if(utils::one_of(perf[i].algorithm,
                                miopenConvolutionAlgoGEMM,
                                miopenConvolutionAlgoDirect,
                                miopenConvolutionAlgoImplicitGEMM)){
                        utils::downcast<miopen_convolution_fwd_pd_t *>(pd)
                            ->set_alg_kind(dnnl_convolution_direct);
                    }
                    else{
                        utils::downcast<miopen_convolution_fwd_pd_t *>(pd)
                            ->set_alg_kind(dnnl_convolution_winograd);
                    }
                    break;
                case dnnl_convolution_direct:
                    if (!utils::one_of(perf[i].algorithm,
                                miopenConvolutionAlgoGEMM,
                                miopenConvolutionAlgoDirect,
                                miopenConvolutionAlgoImplicitGEMM))
                        continue;
                    break;
                case dnnl_convolution_winograd:
                    if (perf[i].algorithm != miopenConvolutionAlgoWinograd)
                        continue;
                    break;
                default: return status::unimplemented;
            }
            
            if(perf[i].algorithm == miopenConvolutionAlgoFFT)
                continue; // just in case
            
            fwd_solution = perf[i];
            // break;
        }

        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenCreateActivationDescriptor, &activation_desc));
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetActivationDescriptor,
                activation_desc,
                miopenActivationMode_t::miopenActivationPASTHRU,
                1.0, 0, 0));
        return status::success;
    }

    // called by configure_post_ops()
    status_t create_and_set_eltwise_descriptor(const convolution_pd_t *pd) {

        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenCreateActivationDescriptor, &eltwise_desc));

        miopenActivationMode_t act_mode;
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
            case alg_kind::eltwise_soft_relu:
                act_mode = miopenActivationSOFTRELU;
                break;
            case alg_kind::eltwise_abs:
                act_mode = miopenActivationABS;
                break;
            default: return status::unimplemented;
        }
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetActivationDescriptor, eltwise_desc,
                act_mode, eltwise_alpha(pd), eltwise_beta(pd), 0));

        return status::success;
    }

    // needed by create_and_set_eltwise_descriptor()
    dnnl::impl::alg_kind_t eltwise_algorithm_kind(const convolution_pd_t *pd) const {
        const int eltwise_idx
                = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return pd->attr()->post_ops_.entry_[eltwise_idx].eltwise.alg;
    }
    float eltwise_alpha(const convolution_pd_t *pd) const {
        const int eltwise_idx
                = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return pd->attr()->post_ops_.entry_[eltwise_idx].eltwise.alpha;
    }
    float eltwise_beta(const convolution_pd_t *pd) const {
        const int eltwise_idx
                = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return pd->attr()->post_ops_.entry_[eltwise_idx].eltwise.beta;
    }
};

struct miopen_convolution_impl_bwd_data_t
    : public miopen_convolution_impl_base_t {
protected:
    miopenConvSolution_t bd_solution;
    std::vector<miopenConvSolution_t> perf;
    size_t requested_algo_count = 0;
    size_t returned_algo_count = 0;
    status_t configure_alg_kind(
            engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
        hip_sycl_scoped_context_handler_t sc(sycl_engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto hip_stream = utils::downcast<sycl_hip_stream_t *>(service_stream);
        auto handle = hip_stream->get_miopen_handle();

        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenConvolutionBackwardDataGetSolutionCount, handle,
                descs[y], weights_desc, conv_desc, descs[x],
                &requested_algo_count));
        perf.resize(requested_algo_count);
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenConvolutionBackwardDataGetSolution,
                handle, descs[y], weights_desc, conv_desc, descs[x],
                requested_algo_count, &returned_algo_count, perf.data()));
        
        // printf("useable algo:%d\n", returned_algo_count);
        // bool algo_selected = false;
        // for(int i = 0; i<returned_algo_count; i++){
        //     if((int)perf[i].algorithm == 1){
        //         bd_solution = perf[i];
        //         algo_selected = true;
        //     }
        // }
        // if(!algo_selected)
        // {
        //     printf("direct conv not avilable!\n");
        //     return status::unimplemented;
        // }

        for (size_t i = 0; i < returned_algo_count; i++) {
            switch (pd->desc()->alg_kind) {
                case dnnl_convolution_auto:
                    if (utils::one_of(perf[i].algorithm,
                                miopenConvolutionAlgoGEMM,
                                miopenConvolutionAlgoDirect,
                                miopenConvolutionAlgoImplicitGEMM)) {
                        utils::downcast<miopen_convolution_bwd_data_pd_t *>(
                                pd)
                                ->set_alg_kind(dnnl_convolution_direct);
                    } else {
                        utils::downcast<miopen_convolution_bwd_data_pd_t *>(
                                pd)
                                ->set_alg_kind(dnnl_convolution_winograd);
                    }
                    break;
                case dnnl_convolution_direct:
                    if (!utils::one_of(perf[i].algorithm,
                                miopenConvolutionAlgoGEMM,
                                miopenConvolutionAlgoDirect,
                                miopenConvolutionAlgoImplicitGEMM))
                        continue;
                    break;
                case dnnl_convolution_winograd:
                    if (perf[i].algorithm != miopenConvolutionAlgoWinograd)
                        continue;
                    break;
                default: return status::unimplemented;
            }

            if(perf[i].algorithm == miopenConvolutionAlgoFFT)
                continue; // just in case

            bd_solution = perf[i];
            break;
        }
        return status::success;
    }

    status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto hip_stream
                = utils::downcast<sycl_hip_stream_t *>(service_stream);
        auto handle = hip_stream->get_miopen_handle();

        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenConvolutionBackwardDataGetWorkSpaceSize,
                handle, descs[io::y], weights_desc, conv_desc, descs[io::x],
                &scratchpad_size));
        if (scratchpad_size > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_cudnn_algo,
                    scratchpad_size, size_t(1));

        return miopen_convolution_impl_base_t::init_scratchpad(engine, pd);
    }

    void execute(miopenHandle_t handle,
            const std::vector<void *> &args) const override {
        auto x = args[0], weights = args[1], y = args[2], bias = args[3],
             scratchpad = args[4];
        if (using_transformed_filter()) {
            auto w_scratch = args[5];
            transform_filter(handle, weights, w_scratch);
            weights = w_scratch;
        }
        MIOPEN_EXECUTE_FUNC_V(miopenConvolutionBackwardDataImmediate, handle,
                descs[io::y], y, weights_desc, weights, conv_desc, descs[io::x], x, 
                scratchpad, scratchpad_size, bd_solution.solution_id);
    }
};

struct miopen_convolution_impl_bwd_weights_t
    : public miopen_convolution_impl_base_t {
protected:
    miopenConvSolution_t bw_solution;
    std::vector<miopenConvSolution_t> perf;
    size_t requested_algo_count = 0;
    size_t returned_algo_count = 0;

public:
    status_t init_zero_dims(convolution_pd_t *pd) override {
        if (pd->ndims() > 8) { return status::invalid_arguments; }
        dnnl_descs[weights] = *pd->invariant_wei_md();
        CHECK(verify_format(&dnnl_descs[weights], true));
        ndims[y] = pd->invariant_dst_md()->ndims;
        ndims[weights] = dnnl_descs[weights].ndims - pd->with_groups();
        CHECK(convert_data_type(&dnnl_descs[weights], &data_types[weights]));
        convert_dims(dnnl_descs[weights].dims + pd->with_groups(),
                dims[weights], ndims[weights]);
        ndims[weights] = std::max(4, ndims[weights]);
        convert_dims(dnnl_descs[weights].format_desc.blocking.strides,
                strides[weights], ndims[weights]);
        CHECK(create_and_set_tensor_descriptor(&descs[weights],
                data_types[weights], ndims[weights], dims[weights],
                strides[weights]));

        if (pd->with_bias()) {
            dnnl_descs[bias] = *pd->invariant_bia_md();
            ndims[bias] = dnnl_descs[bias].ndims;
            CHECK(convert_data_type(&dnnl_descs[bias], &data_types[bias]));
            convert_dims(dnnl_descs[bias].padded_dims, dims[bias], ndims[bias],
                    ndims[y]);
            std::swap(dims[bias][0], dims[bias][1]);
            convert_dims(dnnl_descs[bias].format_desc.blocking.strides,
                    strides[bias], ndims[bias], ndims[weights]);
            ndims[bias] = ndims[y];
            CHECK(create_and_set_tensor_descriptor(&descs[bias],
                    data_types[bias], ndims[bias], dims[bias], strides[bias]));
        }
        return status::success;
    }

    virtual status_t configure_alg_kind(
            engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
        hip_sycl_scoped_context_handler_t sc(sycl_engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto hip_stream
                = utils::downcast<sycl_hip_stream_t *>(service_stream);
        auto handle = hip_stream->get_miopen_handle();

        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenConvolutionBackwardWeightsGetSolutionCount, handle,
                descs[y], descs[x], conv_desc, weights_desc, 
                &requested_algo_count));
        perf.resize(requested_algo_count);
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenConvolutionBackwardWeightsGetSolution,
                handle, descs[y], descs[x], conv_desc, weights_desc,
                requested_algo_count, &returned_algo_count, perf.data()));
        
        // printf("useable algo:%d\n", returned_algo_count);
        // bool algo_selected = false;
        // for(int i = 0; i<returned_algo_count; i++){
        //     if((int)perf[i].algorithm == 1){
        //         bw_solution = perf[i];
        //         algo_selected = true;
        //     }
        // }
        // if(!algo_selected)
        // {
        //     printf("direct conv not avilable!\n");
        //     return status::unimplemented;
        // }

        for (size_t i = 0; i < returned_algo_count; i++) {
            switch (pd->desc()->alg_kind) {
                case dnnl_convolution_auto:
                    if (utils::one_of(perf[i].algorithm,
                                miopenConvolutionAlgoGEMM,
                                miopenConvolutionAlgoDirect,
                                miopenConvolutionAlgoImplicitGEMM)) {
                        utils::downcast<
                                miopen_convolution_bwd_weights_pd_t *>(pd)
                                ->set_alg_kind(dnnl_convolution_direct);
                    } else {
                        utils::downcast<
                                miopen_convolution_bwd_weights_pd_t *>(pd)
                                ->set_alg_kind(dnnl_convolution_winograd);
                    }
                    break;
                case dnnl_convolution_direct:
                    if (!utils::one_of(perf[i].algorithm,
                                miopenConvolutionAlgoGEMM,
                                miopenConvolutionAlgoDirect,
                                miopenConvolutionAlgoImplicitGEMM))
                        continue;
                    break;
                case dnnl_convolution_winograd:
                    if (perf[i].algorithm != miopenConvolutionAlgoWinograd)
                        continue;
                    break;
                default: return status::unimplemented;
            }

            if(perf[i].algorithm == miopenConvolutionAlgoFFT)
                continue; // just in case
            bw_solution = perf[i];
            break;
        }

        return status::success;
    }

    status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto hip_stream
                = utils::downcast<sycl_hip_stream_t *>(service_stream);
        auto handle = hip_stream->get_miopen_handle();

        // it seems miopen set workspacesize in spite of the choosen algorithm
        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenConvolutionBackwardWeightsGetWorkSpaceSize, handle,
                descs[io::y], descs[io::x], conv_desc, weights_desc, &scratchpad_size)); 
        if (scratchpad_size > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_cudnn_algo,
                    scratchpad_size, size_t(1)); // let's just use it

        return miopen_convolution_impl_base_t::init_scratchpad(engine, pd);
    }

    void execute(miopenHandle_t handle,
            const std::vector<void *> &args) const override {
        auto x = args[0], weights = args[1], y = args[2], bias = args[3],
             scratchpad = args[4];
        auto filter = weights;
        if (using_transformed_filter()) {
            auto w_scratch = args[5];
            transform_filter(handle, weights, w_scratch);
            filter = w_scratch;
        }
        const float bias_alpha = 1.0f;
        const float bias_beta = 0.0f;
        MIOPEN_EXECUTE_FUNC_V(miopenConvolutionBackwardWeightsImmediate, handle,
                descs[io::y], y, descs[io::x], x, conv_desc, weights_desc, filter,
                scratchpad, scratchpad_size, bw_solution.solution_id);
        if (with_bias) {
            MIOPEN_EXECUTE_FUNC_V(miopenConvolutionBackwardBias, handle,
                    &bias_alpha, descs[io::y], y, &bias_beta, descs[io::bias],
                    bias);
        }
        if (using_transformed_filter()) {
            undo_transform_filter(handle, filter, weights);
        }
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
