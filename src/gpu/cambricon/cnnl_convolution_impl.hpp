#ifndef GPU_CAMBRICON_CNNL_CONVOLUTION_IMPL_HPP
#define GPU_CAMBRICON_CNNL_CONVOLUTION_IMPL_HPP

#include "cnnl.h"

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/utils.hpp"
#include "gpu/cambricon/cnnl_conv_filter_adjustment_base.hpp"
#include "gpu/cambricon/cnnl_convolution_pd.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_scoped_context.hpp"
#include "gpu/cambricon/sycl_bang_stream.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"

#include <iostream>

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_convolution_impl_base_t
    : public cnnl_conv_filter_adjustment_base_t {
protected:
    enum io { x = 0, bias, weights, y, NUM_IO };
    memory_desc_t dnnl_descs[NUM_IO];
    cnnlConvolutionDescriptor_t conv_desc;
    int padding[CNNL_DIM_MAX];
    int dilation[CNNL_DIM_MAX];
    cnnlTensorDescriptor_t descs[NUM_IO];
    cnnlTensorDescriptor_t quantized_src_desc;
    cnnlTensorDescriptor_t quantized_weight_desc;
    cnnlTensorDescriptor_t quantized_dst_desc;
    cnnlDataType_t quantized_dtype = CNNL_DTYPE_INT16;
    cnnlDataType_t data_types[NUM_IO];
    int ndims[NUM_IO];
    int dims[NUM_IO][DNNL_MAX_NDIMS];
    int strides[NUM_IO + 1][DNNL_MAX_NDIMS];
    int filter_strides[DNNL_MAX_NDIMS];
    cnnlTensorLayout_t formats[NUM_IO];
    bool filter_needs_transform = false;
    float alpha = 0.f;
    float beta = 0.f;
    int group_count = 1;
    bool with_groups = false;
    bool with_bias = false;
    // sratchpad size for some api that need workspace
    size_t scratchpad_size = 0;
    size_t scratchpad_size_qA = 0;
    size_t scratchpad_size_qB = 0;
    size_t scratchpad_size_conv = 0;
    size_t scratchpad_size_bias = 0;    // for backward

    bool use_temp_dst_ = false;
    cnnlDataType_t computation_data_type = CNNL_DTYPE_FLOAT;

public:
    virtual ~cnnl_convolution_impl_base_t() {
        CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, descs[io::weights]);
        CNNL_EXECUTE_FUNC_V(cnnlDestroyConvolutionDescriptor, conv_desc);
        CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, descs[x]);
        CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, descs[y]);
        if(with_bias)
            CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, descs[bias]);
    }
    
    virtual status_t configure_alg_kind(engine_t *, convolution_pd_t *pd) = 0;

    virtual bool supported_filter_format(const memory_desc_t *md) const override {
        const memory_desc_wrapper mem_wrapper(md);
        if(with_groups)
            return mem_wrapper.matches_one_of_tag(format_tag::gohwi, format_tag::goihw);
        else
            return mem_wrapper.matches_one_of_tag(
                    format_tag::acdb,   // NHWC
                    format_tag::cdba,   // HWCN
                    format_tag::ndhwc   // NDHWC
                );
    }

    bool using_transformed_filter() const { return filter_needs_transform; }
    bool with_scratchpad() const { return scratchpad_size > 0; }

    virtual status_t init(engine_t *engine, convolution_pd_t *pd, bool use_scratch_dst = false) {
        CHECK(configure_parameters(pd));
        CHECK(create_cnnl_descs(pd));
        CHECK(check_output_dims());
        CHECK(configure_alg_kind(engine, pd));
        CHECK(init_scratchpad(engine, pd));

        return status::success;
    }

    virtual status_t init_zero_dims(convolution_pd_t *pd) {
        return status::success;
    }
    
    // cnnl seems don't support strided data storage
    void get_dims_and_strides(int io) {
        convert_dims(dnnl_descs[io].dims, dims[io], dnnl_descs[io].ndims, ndims[io]);
        // TODO: assert(stride == 1)
    }

    status_t configure_parameters(const convolution_pd_t *pd) {
        if (pd->ndims() > CNNL_DIM_MAX) { return status::invalid_arguments; }
        CHECK(set_padding_and_dilation(pd));
        with_groups = pd->with_groups();
        with_bias = pd->with_bias();
        alpha = 1.0f;
        beta = 0.0f;

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

        CHECK(get_formats());
        set_compute_format();
        get_dims_and_strides(x);
        get_dims_and_strides(weights);
        get_dims_and_strides(y);

        if (!supported_filter_format(&dnnl_descs[weights])) {
            // TODO: transform(transpose) filter format into supported layout
            assert(0 && "not supported format!");
            set_filter_format(ndims[weights], dims[weights], strides[NUM_IO], formats[x]);
            CHECK(init_filter_transformation(data_types[weights],
                    ndims[weights], dims[weights], strides[weights],
                    strides[NUM_IO]));
            filter_needs_transform = true;
            // we transform the filter based on src format
            formats[weights] = formats[x];
        } else {
            CHECK(get_filter_format());
            // get_dims_and_strides(weights);
        }

        // TODO: depth-wise convolution should be considered
        if (with_groups) {
            // Group convolution
            // dnnl takes (group, OutC/group, kH, kW, InC/group) as filter's shape(gohwi) in group convolution
            // but cnnl takes (OutC, kH, kW, InC/group) as filter's shape(ohwi), and group_count as an extra parameter to set convdesc
            dims[weights][1] *= pd->G();
            ndims[weights] = std::max(4, ndims[weights] - with_groups);
        }
        if (with_bias) {
            // cudnn's conv kernel don't fuse bias by default, cnnl does, so there is no need to convert bias dim
            ndims[bias] = dnnl_descs[bias].ndims;
            CHECK(convert_data_type(&dnnl_descs[bias], &data_types[bias]));
            convert_dnnl_dims_array(dnnl_descs[bias].dims, dims[bias], ndims[bias]);
            // convert_dims(dnnl_descs[bias].dims, dims[bias], ndims[bias], ndims[y]);
            // std::swap(dims[bias][0], dims[bias][1]);
            // convert_dims(dnnl_descs[bias].format_desc.blocking.strides, strides[bias], ndims[bias], ndims[y]);
            // ndims[bias] = ndims[y];
        }

        return status::success;
    }

    status_t create_cnnl_descs(const convolution_pd_t *pd) {
        CHECK(create_and_set_convolution_desc(pd));
        // Here, format[x] and format[y] should be NHWC
        CHECK(create_and_set_tensor_descriptor(&descs[x], formats[x], data_types[x], ndims[x], dims[x]));
        CHECK(create_and_set_tensor_descriptor(&quantized_src_desc, formats[x], quantized_dtype, ndims[x], dims[x]));

        CHECK(create_and_set_tensor_descriptor(&descs[y], formats[y], data_types[y], ndims[y], dims[y]));
        CHECK(create_and_set_tensor_descriptor(&quantized_dst_desc, formats[y], quantized_dtype, ndims[y], dims[y]));

        CHECK(create_and_set_tensor_descriptor(&descs[weights], formats[weights], data_types[weights], ndims[weights], dims[weights] + with_groups));
        CHECK(create_and_set_tensor_descriptor(&quantized_weight_desc, formats[weights], quantized_dtype, ndims[weights], dims[weights] + with_groups));

        if (with_bias) {
            assert(ndims[bias] == 1);
            CHECK(create_and_set_tensor_descriptor(&descs[bias], CNNL_LAYOUT_ARRAY, data_types[y], ndims[bias], dims[bias]));
        }
        else {
            descs[bias] = nullptr;
            // int bias_dim[1] = {dims[y][3]};
            // CHECK(create_and_set_tensor_descriptor(&descs[bias], CNNL_LAYOUT_ARRAY, data_types[y], 1, bias_dim));
        }

        return status::success;
    }
    
    // init scratchpad for filter transformation
    virtual status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) {
        // TODO: check filter transform
        assert(!filter_needs_transform);
        if (filter_needs_transform) {
            auto sz = memory_desc_wrapper(&dnnl_descs[weights]).size();
            auto data_size = types::data_type_size(pd->invariant_wei_md(0)->data_type);
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_cnnl, sz,
                    data_size);
        }
        return status::success;
    };

    status_t create_and_set_convolution_desc(const convolution_pd_t *pd) {
        CNNL_EXECUTE_FUNC_V(cnnlCreateConvolutionDescriptor, &conv_desc);

        assert(ndims[x] == 4 || ndims[x] == 5);  // for debug
        // cnnl only support cross_corrrelation. and this is a "group count = 1" conv(no depth-wise)
        CNNL_EXECUTE_FUNC_V(cnnlSetConvolutionDescriptor, conv_desc,
                ndims[x], padding, filter_strides, dilation,
                1, computation_data_type);

        if (with_groups) {
            group_count = pd->G();
            if (group_count > 1){
                CNNL_EXECUTE_FUNC_V(cnnlSetConvolutionDescriptor, conv_desc,
                    ndims[x], padding, filter_strides, dilation,
                    group_count, computation_data_type);
            }
        }
        return status::success;
    }

    // TODO: match cnnl and onednn
    status_t set_padding_and_dilation(const convolution_pd_t *pd) {
        int actual_ndims = pd->ndims();
        if (actual_ndims == 4) {
            padding[0] = static_cast<int>(pd->padT());
            padding[1] = static_cast<int>(pd->padT());
            padding[2] = static_cast<int>(pd->padL());
            padding[3] = static_cast<int>(pd->padL());

            // why????
            dilation[0] = static_cast<int>(pd->KDH());
            dilation[1] = static_cast<int>(pd->KDW());

            filter_strides[0] = static_cast<int>(pd->KSH());
            filter_strides[1] = static_cast<int>(pd->KSW());
        } else {
            padding[0] = static_cast<int>(pd->padFront());
            padding[1] = static_cast<int>(pd->padFront());
            padding[2] = static_cast<int>(pd->padT());
            padding[0] = static_cast<int>(pd->padT());
            padding[1] = static_cast<int>(pd->padL());
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

    virtual void execute(cnnlHandle_t handle, const std::vector<void *> &args) const = 0;

    // sum - kind post operator(constant layer?)
    void execute_sum(cnnlHandle_t handle, void *x, void *y, float alpha_, float beta_) const {
        assert(alpha_ == 1.0f && beta_ == 1.0f);    // TODO: do scaled add
        void* addn_srcs[2] = {x, y};
        cnnlTensorDescriptor_t addn_descs[2] = {descs[io::y], descs[io::y]};
        CNNL_EXECUTE_FUNC_V(cnnlAddN, handle, addn_descs, addn_srcs, 2, descs[io::y], y);
    }

    // TODO: zero dims need this?
    void execute_set_weights_bias(cnnlHandle_t handle, void *weights, void *bias, float value){
        return;
    }

    bool with_eltwise(const convolution_pd_t *pd, int position) const {
        return pd->attr()->post_ops_.contain(primitive_kind::eltwise, position);
    }

    status_t check_output_dims() const {
        int expected_dims[CNNL_DIM_MAX] = {};
        CNNL_EXECUTE_FUNC_V(cnnlGetConvolutionForwardOutputDim, conv_desc,
                descs[x], descs[io::weights], ndims[y], &expected_dims[0]);
        for (size_t i = 0; i < ndims[y]; i++) {
            if (dims[y][i] != expected_dims[i]) return status::unimplemented;
        }
        return status::success;
    }

    void set_compute_format() {
        computation_data_type = CNNL_DTYPE_FLOAT;
    }

    status_t get_filter_format() {
        memory_desc_wrapper wrapper(&dnnl_descs[weights]);
        // consider gohwi as (g*o, h, w, i)
        if (wrapper.matches_one_of_tag(format_tag::acdb, format_tag::gohwi, format_tag::goihw)) {
            formats[weights] =  cnnlTensorLayout_t::CNNL_LAYOUT_NHWC;
        } else if (wrapper.matches_one_of_tag(format_tag::cdba)) {
            formats[weights] =  cnnlTensorLayout_t::CNNL_LAYOUT_HWCN;
        } else if (wrapper.matches_one_of_tag(format_tag::ndhwc)){
            formats[weights] =  cnnlTensorLayout_t::CNNL_LAYOUT_NDHWC;
        } else {
            return status::unimplemented;            
        }
        return status::success;
    }

    status_t get_formats() {
        CHECK(get_format(&dnnl_descs[x], formats[x]));
        CHECK(get_format(&dnnl_descs[y], formats[y]));
        return status::success;
    }

    void set_filter_nhwc(int filter_ndims, int *transform_filter_strides,
            int *filter_dims) override {
        if (with_groups) {
            switch (filter_ndims) {
                case 4: // Convert to krsc
                    return propagate_strides(transform_filter_strides,
                            filter_dims, {2, 3, 1, 0});
                case 5:
                    return propagate_strides(transform_filter_strides,
                            filter_dims, {2, 4, 3, 1, 0});
            }
        } else {
            cnnl_conv_filter_adjustment_base_t::set_filter_nhwc(filter_ndims, transform_filter_strides, filter_dims);
        }
    }

    bool use_temp_dst() const { return use_temp_dst_; }
};

struct cnnl_convolution_impl_fwd_t : public cnnl_convolution_impl_base_t {
protected:
    cnnlTensorDescriptor_t reorder_dst_desc = nullptr;
    cnnlConvolutionForwardAlgo_t fwd_alg_kind = CNNL_CONVOLUTION_FWD_ALGO_DIRECT; // CNNL_CONVOLUTION_FWD_ALGO_DIRECT CNNL_CONVOLUTION_FWD_ALGO_GEMM
    cnnlConvolutionFwdPreference_t cnnl_perf = CNNL_CONVOLUTION_FWD_FASTEST;    // only this was implemented
    int supported_algo_count = 0;
    int num_post_ops = 0;
    primitive_kind_t post_ops[2];
    bool need_reorder = false;
    float sum_scale = 1.0f;
    bool conv_bias_eltwise = false;
    bool conv_bias = false;
    
    // fusioned convolution forward
    cnnlFusedOps_t fusion_type = CNNL_CONV_SCALE_BN_ACTIVATION;
    cnnlFusedOpsPlan_t fusion_plan = nullptr;
    cnnlFusedOpsConstParamPack_t cparam_pack = nullptr;
    cnnlFusedOpsVariantParamPack_t vparam_pack = nullptr;
    cnnlConvolutionCastMode_t cast_mode;

    bool do_scaling = false;
    cnnlTensorDescriptor_t scale_alpha_desc, scale_beta_desc;
    float output_scaling = 1.0f;
    float output_offsets = 0.f;
    void *scale_alpha, *scale_beta;

    bool do_batchnorm = false;
    cnnlTensorDescriptor_t wbmv_desc;
    float bn_eps = 0.0001f;
    void *bn_weight, *bn_bias;
    void *e_mean, *e_var;

    bool do_activ = false;
    cnnlActivationDescriptor_t activation_desc;
    float activ_alpha[1] = {1.0f};
    float activ_beta=0.f, activ_coef=1.0f;

    size_t fu_workspace_size = 0;
public:
    virtual ~cnnl_convolution_impl_fwd_t() {
        if (activation_desc)
            CNNL_EXECUTE_FUNC_V(
                    cnnlDestroyActivationDescriptor, activation_desc);
        if (reorder_dst_desc)
            CNNL_EXECUTE_FUNC_V(
                    cnnlDestroyTensorDescriptor, reorder_dst_desc);
        // TODO: free some extra cnrt pointers
    }

    status_t configure_post_ops(convolution_pd_t *pd) {
        // scale, TODO: add vector support
        output_scaling = pd->attr()->output_scales_.scales_[0];
        output_offsets = pd->attr()->output_offsets_.offsets_[0];
        do_scaling = (output_scaling != 1.f) || (output_offsets != 0.f);

        // bn, eltwise, sum ...
        auto &p = pd->attr()->post_ops_;
        num_post_ops = p.len();
        for (size_t i = 0; i < p.len(); i++) {
            post_ops[i] = p.entry_[i].kind;
            if (post_ops[i] == dnnl_eltwise) {
                do_activ = true;
                CHECK(create_and_set_activation_descriptor(pd));
            }
            if (post_ops[i] == dnnl_batch_normalization) {
                do_batchnorm = true;
                bn_eps = p.entry_[i].batchnorm.epsilon;
            }
            if (post_ops[i] == dnnl_sum) {
                sum_scale = p.entry_[i].sum.scale; 
            }
        }
        return status::success;
    }

    status_t init(engine_t *engine, convolution_pd_t *pd,
            bool use_scratch_dst) override {
        use_temp_dst_ = use_scratch_dst;
        CHECK(configure_parameters(pd));
        CHECK(configure_post_ops(pd));
        CHECK(create_cnnl_descs(pd));
        CHECK(configure_alg_kind(engine, pd));
        CHECK(init_scratchpad(engine, pd));

        return status::success;
    }

    // TODO: layout transform, this won't work now
    void execute_reorder(cnnlHandle_t handle, void *src, void *dst,
            bool flip_formats) const {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        // derive permute
        int trans_permute[4] = {0, 1, 2, 3};
        cnnlTransposeDescriptor_t trans_desc;
        CNNL_EXECUTE_FUNC_V(cnnlCreateTransposeDescriptor, &trans_desc);
        CNNL_EXECUTE_FUNC_V(cnnlSetTransposeDescriptor, trans_desc, 4, trans_permute);

        if (flip_formats) {
            CNNL_EXECUTE_FUNC_V(cnnlTranspose, handle, trans_desc,
                    reorder_dst_desc, src, descs[y], dst);
        } else {
            CNNL_EXECUTE_FUNC_V(cnnlTranspose, handle, trans_desc, 
                    descs[y], src, reorder_dst_desc, dst);
        }
        CNNL_EXECUTE_FUNC_V(cnnlDestroyTransposeDescriptor, trans_desc);
    }

    void execute_eltwise(cnnlHandle_t handle, void *src, void *dst) const {
        float alpha = 1.0f;
        float beta = 0.0f;
        CNNL_EXECUTE_FUNC_V(cnnlActivationForward, handle, activation_desc,
                &alpha, descs[io::y], src, &beta, descs[io::y], dst);
    }

    void execute(cnnlHandle_t handle, const std::vector<void *> &args) const override {
        auto start = std::chrono::steady_clock::now();
        
        auto x = args[0], weights = args[1], y = args[2], bias = args[3],
             scratchpad = args[4], post_op_scratch = args[6],
             post_op_reorder = args[7];

        assert(!use_temp_dst_);
        void *output = use_temp_dst_ ? post_op_scratch : y;
        // not support yet!
        // if (using_transformed_filter()) {
        //     auto w_scratch = args[5];
        //     transform_filter(handle, weights, w_scratch);
        //     weights = w_scratch;
        // }

        // quantization
        // These quantized tensors should be saved for backward, but not implemented yet
        void *d_q_src, *d_q_wei;
        int src_size = dims[io::x][0]*dims[io::x][1]*dims[io::x][2]*dims[io::x][3];
        // consider group weight
        int weight_size = dims[io::weights][0]*dims[io::weights][1]*dims[io::weights][2]*dims[io::weights][3];
        if(with_groups)
            weight_size *= dims[io::weights][4];
        assert(src_size>0 && weight_size>0);
        auto err0 = cnrtMalloc(&d_q_src, sizeof(int16_t) * src_size);
        auto err1 = cnrtMalloc(&d_q_wei, sizeof(int16_t) * weight_size);
        if(err0 != CNNL_STATUS_SUCCESS || err1 != CNNL_STATUS_SUCCESS) 
            assert(0 && "err when cnrtmalloc");
        
        cnrtMemset(d_q_src, 0, sizeof(int16_t) * src_size);
        cnrtMemset(d_q_wei, 0, sizeof(int16_t) * weight_size);
        cnrtSyncDevice();

        // Quantize input
        void* scratchpad_qA = scratchpad_size_qA > 0 ? scratchpad : nullptr;
        quantize_array(handle, descs[io::x], x, 16, scratchpad_qA, scratchpad_size_qA, quantized_src_desc, d_q_src);
        // Quantize weight
        void* scratchpad_qB = scratchpad_size_qB > 0 ? scratchpad : nullptr;
        quantize_array(handle, descs[io::weights], weights, 16, scratchpad_qB, scratchpad_size_qB, quantized_weight_desc, d_q_wei);
        
        void* d_bias = bias;

        bool fused = do_scaling || do_batchnorm || do_activ;
        // cnnl fuse conv and bias by default 
        if (fused && group_count == 1) {
            // var pack
            if(fu_workspace_size > 0)
                CNNL_EXECUTE_FUNC_V(cnnlSetFusedOpsVariantParamPackAttribute, vparam_pack, CNNL_PTR_WORKSPACE, scratchpad);
            CNNL_EXECUTE_FUNC_V(cnnlSetFusedOpsVariantParamPackAttribute, vparam_pack, CNNL_SCALAR_WORKSPACE_SIZE, (void*)(&fu_workspace_size));
            // conv
            CNNL_EXECUTE_FUNC_V(cnnlSetFusedOpsVariantParamPackAttribute, vparam_pack, CNNL_PTR_X, (void*)(d_q_src));
            CNNL_EXECUTE_FUNC_V(cnnlSetFusedOpsVariantParamPackAttribute, vparam_pack, CNNL_PTR_W, (void*)(d_q_wei));
            CNNL_EXECUTE_FUNC_V(cnnlSetFusedOpsVariantParamPackAttribute, vparam_pack, CNNL_PTR_Y, (void*)(output));
            CNNL_EXECUTE_FUNC_V(cnnlSetFusedOpsVariantParamPackAttribute, vparam_pack, CNNL_PTR_BIAS, (void*)(bias));
            if(do_batchnorm){
                CNNL_EXECUTE_FUNC_V(cnnlSetFusedOpsVariantParamPackAttribute, vparam_pack, CNNL_PTR_BN_MEAN, (void*)(e_mean));
                CNNL_EXECUTE_FUNC_V(cnnlSetFusedOpsVariantParamPackAttribute, vparam_pack, CNNL_PTR_BN_VAR, (void*)(e_var));
                CNNL_EXECUTE_FUNC_V(cnnlSetFusedOpsVariantParamPackAttribute, vparam_pack, CNNL_PTR_BN_WEIGHT, (void*)bn_weight);
                CNNL_EXECUTE_FUNC_V(cnnlSetFusedOpsVariantParamPackAttribute, vparam_pack, CNNL_PTR_BN_BIAS, (void*)bn_bias);
                CNNL_EXECUTE_FUNC_V(cnnlSetFusedOpsVariantParamPackAttribute, vparam_pack, CNNL_SCALAR_BN_EPSILON, (void*)(&bn_eps));                
            }
            if(do_scaling){
                CNNL_EXECUTE_FUNC_V(cnnlSetFusedOpsVariantParamPackAttribute, vparam_pack, CNNL_PTR_SCALE_ALPHA, (void*)(scale_alpha));
                CNNL_EXECUTE_FUNC_V(cnnlSetFusedOpsVariantParamPackAttribute, vparam_pack, CNNL_PTR_SCALE_BETA, (void*)(scale_beta));
            }
            cnrtSyncDevice();
            // assert(CNNL_OFFLINE_SYMMETRIC_QUANTIZE == cast_mode);
            CNNL_EXECUTE_FUNC_V(cnnlFusedOpsExecute, handle, fusion_plan, vparam_pack);
        } 
        else { // fwd_alg_kind CNNL_CONVOLUTION_FWD_ALGO_DIRECT CNNL_CONVOLUTION_FWD_ALGO_GEMM
            void* scratchpad_conv = scratchpad_size_conv > 0 ? scratchpad : nullptr;
            CNNL_EXECUTE_FUNC_V(cnnlConvolutionForward, handle, conv_desc, fwd_alg_kind, nullptr,
                quantized_src_desc, d_q_src, quantized_weight_desc, d_q_wei, descs[io::bias], d_bias, 
                scratchpad_conv, scratchpad_size_conv, nullptr, descs[io::y], output);
            
            if(do_scaling){
                CNNL_EXECUTE_FUNC_V(cnnlScale, handle, 0, descs[io::y], output, scale_alpha_desc, scale_alpha,
                                    scale_beta_desc, scale_beta, descs[io::y], output);
            }
            if(do_batchnorm){
                CNNL_EXECUTE_FUNC_V(cnnlBatchNormForwardInference, handle, NULL, NULL, descs[io::y], output,
                                    wbmv_desc, bn_weight, bn_bias, e_mean, e_var, bn_eps, descs[io::y], output);
            }
            if(do_activ){
                CNNL_EXECUTE_FUNC_V(cnnlActivationForward, handle, activation_desc, &activ_alpha[0],
                                    descs[io::y], output, &activ_beta, descs[io::y], output);
            }
        }

        // other post-op that can't be fused at kernel layer
        for (int i = 0; i < num_post_ops; i++) {
            bool last_op = i == num_post_ops - 1 && !need_reorder;
            switch (post_ops[i]) {
                case dnnl_sum:
                    if (need_reorder) {
                        execute_reorder(handle, y, post_op_reorder, true);
                        execute_sum(handle, post_op_reorder, post_op_scratch, sum_scale, 1.0f);
                    } else if (last_op) {
                        execute_sum(handle, post_op_scratch, y, 1.0f, sum_scale);
                    } else {
                        execute_sum(handle, y, post_op_scratch, sum_scale, 1.0f);
                    }
                    break;
                default: 
                    break;
                    // assert(!"unsupported post op");
            }
        }
        
        if (need_reorder) {
            execute_reorder(handle, post_op_scratch, y, false);
        }

        // free MLU memory space, TODO: change this
        if(0)
        {
            if(d_q_src != nullptr) cnrtFree(d_q_src);
            if(d_q_wei != nullptr) cnrtFree(d_q_wei);
            
            if(bn_weight != nullptr) cnrtFree(bn_weight);
            if(bn_bias != nullptr) cnrtFree(bn_bias);
            if(e_mean != nullptr) cnrtFree(e_mean);
            if(e_var != nullptr) cnrtFree(e_var);
            
            if(scale_alpha != nullptr) cnrtFree(scale_alpha);
            if(scale_beta != nullptr) cnrtFree(scale_beta);
        }
    
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
        float t = float(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
        // printf("src shape:(%d, %d, %d, %d), weight shape:(%d, %d, %d, %d, %d)\n",
        //     dims[io::x][0], dims[io::x][1], dims[io::x][2], dims[io::x][3],
        //     dims[io::weights][0], dims[io::weights][1], dims[io::weights][2], dims[io::weights][3], dims[io::weights][4]);
        // printf("conv kernel time:%f\n", t);
    }
    
    status_t init_fuse_plan(cnnlHandle_t handle){
        // plan (conv based)
        cnnlCreateFusedOpsPlan(&fusion_plan, fusion_type);
        cnnlCreateFusedOpsVariantParamPack(&vparam_pack, fusion_type);
        cnnlCreateFusedOpsConstParamPack(&cparam_pack, fusion_type);
        
        // const pack
        cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_XDESC, (void*)quantized_src_desc);
        cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_WDESC, (void*)quantized_weight_desc);
        cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_YDESC, (void*)descs[io::y]);
        cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_BIAS_DESC, (void*)descs[io::bias]);
        cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_CONV_DESC, (void*)conv_desc);
        cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_SCALAR_CONV_FWD_ALGO, (void*)(&fwd_alg_kind));
        cast_mode = CNNL_OFFLINE_SYMMETRIC_QUANTIZE;
        cnnlSetFusedOpsConstParamPackAttribute(cparam_pack,  CNNL_SCALAR_CONV_FWD_CAST_MODE, (void*)(&cast_mode));
        if(do_scaling){
            cnnlCreateTensorDescriptor(&scale_alpha_desc);
            cnnlCreateTensorDescriptor(&scale_beta_desc);
            auto batch = dims[io::y][0];
            cnnlSetTensorDescriptor(scale_alpha_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 1, dims[io::y]);
            cnnlSetTensorDescriptor(scale_beta_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 1, dims[io::y]);
            // this is ugly, we should try use scratchpad to manage these extra memory need in fusion kernel.
            cnrtMalloc(&scale_alpha, sizeof(float)*batch);
            cnrtMalloc(&scale_beta, sizeof(float)*batch);
            std::vector<float> host_alpha(batch, output_scaling);
            std::vector<float> host_beta(batch, output_offsets);
            cnrtMemcpy(scale_alpha, host_alpha.data(), sizeof(float)*host_alpha.size(), CNRT_MEM_TRANS_DIR_HOST2DEV);
            cnrtMemcpy(scale_beta, host_beta.data(), sizeof(float)*host_beta.size(), CNRT_MEM_TRANS_DIR_HOST2DEV);
            
            cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_SCALE_ALPHA_DESC, (void*)scale_alpha_desc);
            cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_SCALE_BETA_DESC, (void*)scale_beta_desc);                    
        }
        if(do_batchnorm){
            cnnlCreateTensorDescriptor(&wbmv_desc);
            auto output_c = dims[io::y][3];
            cnnlSetTensorDescriptor(wbmv_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 1, &dims[io::y][3]);
            // this is ugly, and bn's weight and bias set to 1/0 will cause wrong result.
            cnrtMalloc(&bn_weight, sizeof(float)*output_c);
            cnrtMalloc(&bn_bias, sizeof(float)*output_c);
            cnrtMemset(bn_weight, 1, sizeof(float)*output_c);
            cnrtMemset(bn_bias, 0, sizeof(float)*output_c);
            cnrtMalloc(&e_mean, sizeof(float));
            cnrtMalloc(&e_var, sizeof(float));
            cnrtMemset(e_mean, 0, sizeof(float));
            cnrtMemset(e_var, 0, sizeof(float));

            cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_BN_WEIGHT_BIAS_MEAN_VAR_DESC, (void*)wbmv_desc);
        }
        if(do_activ){
            // activation_desc is create and set in configure_post_ops() function
            cnnlSetFusedOpsConstParamPackAttribute(cparam_pack, CNNL_ACTIVATION_DESC, (void*)activation_desc);                
        }

        cnnlMakeFusedOpsPlan(handle, fusion_plan, cparam_pack, &fu_workspace_size);

        return status::success;
    }

    // init scratchpad for convolution kernel
    status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto bang_stream = utils::downcast<sycl_bang_stream_t *>(service_stream);
        auto handle = bang_stream->get_cnnl_handle();
        
        // if fusion is needed, fu_workspace_size will be set after this, and conv workspace won't be need any more.
        // fused conv not support group conv
        scratchpad_size = 0;
        if((do_scaling || do_batchnorm || do_activ) && group_count == 1) {
            CHECK(init_fuse_plan(handle));
            scratchpad_size = std::max(fu_workspace_size, scratchpad_size);            
        }
        else {
            CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetConvolutionForwardWorkspaceSize, handle, 
                    quantized_src_desc, quantized_weight_desc, descs[y], descs[bias], 
                    conv_desc, fwd_alg_kind, &scratchpad_size_conv));
            scratchpad_size = std::max(scratchpad_size_conv, scratchpad_size);            
        }

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetQuantizeParamWorkspaceSize, handle, descs[x], &scratchpad_size_qA));
        scratchpad_size = std::max(scratchpad_size_qA, scratchpad_size);

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetQuantizeParamWorkspaceSize, handle, descs[io::weights], &scratchpad_size_qB));
        scratchpad_size = std::max(scratchpad_size_qB, scratchpad_size);

        // TODO: check if "memory_tracking::names" is an exclusive key
        if (scratchpad_size > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_cnnl,
                    scratchpad_size, size_t(1));

        CHECK(cnnl_convolution_impl_base_t::init_scratchpad(engine, pd));
        return status::success;
    }

    status_t configure_alg_kind(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
        bang_sycl_scoped_context_handler_t sc(sycl_engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto bang_stream = utils::downcast<sycl_bang_stream_t *>(service_stream);
        auto handle = bang_stream->get_cnnl_handle();

        supported_algo_count = 2;   // cnnl now only support 2 kind algorithm, and have no count request api 

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetConvolutionForwardAlgorithm, handle, conv_desc,
                descs[x], descs[io::weights], descs[y], cnnl_perf, &fwd_alg_kind));
         
        if(!utils::one_of(fwd_alg_kind, CNNL_CONVOLUTION_FWD_ALGO_DIRECT, CNNL_CONVOLUTION_FWD_ALGO_GEMM))
            return status::unimplemented;
        
        // cnnl not support winograd convolution
        if(pd->desc()->alg_kind == dnnl_convolution_winograd)
            return status::unimplemented;
        
        // dnnl treat gemm as direct convolution
        utils::downcast<cnnl_convolution_fwd_pd_t *>(pd) -> set_alg_kind(dnnl_convolution_direct);

        return status::success;
    }

    status_t create_and_set_activation_descriptor(const convolution_pd_t *pd) {
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlCreateActivationDescriptor, &activation_desc));
        cnnlActivationMode_t act_mode;
        const int eltwise_idx
                = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        dnnl::impl::alg_kind_t algkind = pd->attr()->post_ops_.entry_[eltwise_idx].eltwise.alg;
        switch (algkind) {
            case alg_kind::eltwise_tanh:
                act_mode = CNNL_ACTIVATION_TANH;
                break;
            case alg_kind::eltwise_elu: 
                act_mode = CNNL_ACTIVATION_ELU; 
                break;
            case alg_kind::eltwise_relu:
                act_mode = CNNL_ACTIVATION_RELU;
                break;
            case alg_kind::eltwise_logistic:
                act_mode = CNNL_ACTIVATION_SIGMOID;
                break;
            case alg_kind::eltwise_bounded_relu:
                act_mode = CNNL_ACTIVATION_CLIPPED_RELU;
                break;
            default: return status::unimplemented;
        }
        // ELU use coef and GLU(not implemented) use sliced_dim
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetActivationDescriptor_v3, activation_desc,
                act_mode, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_NOT_PROPAGATE_NAN,
                activ_coef, 0));

        return status::success;
    }
};

struct cnnl_convolution_impl_bwd_data_t
    : public cnnl_convolution_impl_base_t {
protected:
    cnnlConvolutionBwdDataAlgo_t bwd_algo = CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT;
    cnnlConvolutionBwdDataPreference_t cnnl_perf = CNNL_CONVOLUTION_BWD_DATA_FASTEST;   // there is another choice
    int requested_algo_count = 0;
    int returned_algo_count = 0;

    status_t configure_alg_kind(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
        bang_sycl_scoped_context_handler_t sc(sycl_engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto bang_stream = utils::downcast<sycl_bang_stream_t *>(service_stream);
        auto handle = bang_stream->get_cnnl_handle();

        // cnnl only support CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT
        if(pd->desc()->alg_kind == dnnl_convolution_winograd)
            return status::unimplemented;
        utils::downcast<cnnl_convolution_bwd_data_pd_t *>(pd)->set_alg_kind(dnnl_convolution_direct);
        return status::success;
    }

    status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto bang_stream = utils::downcast<sycl_bang_stream_t *>(service_stream);
        auto handle = bang_stream->get_cnnl_handle();
        
        scratchpad_size = 0;
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetConvolutionBackwardDataWorkspaceSize,
                handle, quantized_weight_desc, quantized_dst_desc, conv_desc, descs[io::x],
                bwd_algo, &scratchpad_size_conv));
        scratchpad_size = std::max(scratchpad_size_conv, scratchpad_size);
        
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetQuantizeParamWorkspaceSize, handle, descs[y], &scratchpad_size_qA));
        scratchpad_size = std::max(scratchpad_size_qA, scratchpad_size);

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetQuantizeParamWorkspaceSize, handle, descs[weights], &scratchpad_size_qB));
        scratchpad_size = std::max(scratchpad_size_qB, scratchpad_size);
        
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetBiasAddWorkspaceSize, handle, descs[io::bias], descs[io::x], &scratchpad_size_bias));
        scratchpad_size = std::max(scratchpad_size_bias, scratchpad_size);

        if (scratchpad_size > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_cnnl,
                    scratchpad_size, size_t(1));
        
        return cnnl_convolution_impl_base_t::init_scratchpad(engine, pd);
    }

    void execute(cnnlHandle_t handle, const std::vector<void *> &args) const override {
        auto x = args[0], weights = args[1], y = args[2], bias = args[3],
             scratchpad = args[4];
        if (using_transformed_filter()) {
            auto w_scratch = args[5];
            transform_filter(handle, weights, w_scratch);
            weights = w_scratch;
        }
        const float bias_alpha = 1.0f;
        const float bias_beta = 1.0f;

        // quantization
        // device_quantized_gradient_destination
        void *d_q_g_dst, *d_q_wei;
        int dst_size = dims[io::y][0]*dims[io::y][1]*dims[io::y][2]*dims[io::y][3];
        int weight_size = dims[io::weights][0]*dims[io::weights][1]*dims[io::weights][2]*dims[io::weights][3];
        cnrtMalloc(&d_q_g_dst, sizeof(int16_t) * dst_size);
        cnrtMalloc(&d_q_wei, sizeof(int16_t) * weight_size);
        cnrtMemset(d_q_g_dst, 0, sizeof(int16_t) * dst_size);
        cnrtMemset(d_q_wei, 0, sizeof(int16_t) * weight_size);
        cnrtSyncDevice();

        // Quantize g_y
        void* scratchpad_qA = scratchpad_size_qA > 0 ? scratchpad : nullptr;
        quantize_array(handle, descs[io::y], y, 16, scratchpad_qA, scratchpad_size_qA, quantized_dst_desc, d_q_g_dst);
        // Quantize weight
        void* scratchpad_qB = scratchpad_size_qB > 0 ? scratchpad : nullptr;
        quantize_array(handle, descs[io::weights], weights, 16, scratchpad_qB, scratchpad_size_qB, quantized_weight_desc, d_q_wei);

        void* scratchpad_conv = scratchpad_size_conv > 0 ? scratchpad : nullptr;
        CNNL_EXECUTE_FUNC_V(cnnlConvolutionBackwardData, handle, &alpha,
                quantized_weight_desc, d_q_wei, quantized_dst_desc, d_q_g_dst, conv_desc, bwd_algo,
                scratchpad_conv, scratchpad_size_conv, &beta, descs[io::x], x);
        std::cout<<"cnnl_impl with_bias flag is : "<<with_bias<<std::endl;
        if (with_bias) {
            // TODO: check this, convolution backward have bias?
            void* scratchpad_bias = scratchpad_size_bias > 0 ? scratchpad : nullptr;
            CNNL_EXECUTE_FUNC_V(cnnlBiasAdd, handle, &bias_alpha, descs[io::bias], bias, 
                scratchpad_bias, scratchpad_size_bias, &bias_beta, descs[io::x], x);
        }
    }
};

struct cnnl_convolution_impl_bwd_weights_t
    : public cnnl_convolution_impl_base_t {
protected:
    cnnlConvolutionBwdFilterAlgo_t bwd_filter_algo = CNNL_CONVOLUTION_BWD_FILTER_ALGO_GEMM;
    cnnlConvolutionBwdFilterPreference_t cnnl_perf;
    int supported_algo_count = 0;

public:
    status_t init_zero_dims(convolution_pd_t *pd) override {
        if (pd->ndims() > CNNL_DIM_MAX) { return status::invalid_arguments; }
        dnnl_descs[weights] = *pd->invariant_wei_md();
        CHECK(get_format(&dnnl_descs[weights], formats[weights], true));
        ndims[y] = pd->invariant_dst_md()->ndims;
        ndims[weights] = dnnl_descs[weights].ndims - pd->with_groups();
        CHECK(convert_data_type(&dnnl_descs[weights], &data_types[weights]));
        convert_dims(dnnl_descs[weights].dims + pd->with_groups(),
                dims[weights], ndims[weights]);
        ndims[weights] = std::max(4, ndims[weights]);
        convert_dims(dnnl_descs[weights].format_desc.blocking.strides,
                strides[weights], ndims[weights]);
        CHECK(create_and_set_tensor_descriptor(&descs[weights], CNNL_LAYOUT_NHWC,
                data_types[weights], ndims[weights], dims[weights]));

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
            CHECK(create_and_set_tensor_descriptor(&descs[bias], CNNL_LAYOUT_ARRAY,
                    data_types[bias], ndims[bias], dims[bias]));
        }
        return status::success;
    }
    
    virtual status_t configure_alg_kind(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
        bang_sycl_scoped_context_handler_t sc(sycl_engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto bang_stream = utils::downcast<sycl_bang_stream_t *>(service_stream);
        auto handle = bang_stream->get_cnnl_handle();

        supported_algo_count = 2;

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetConvolutionBackwardFilterAlgorithm,
                handle, conv_desc, descs[x], descs[y], descs[io::weights], cnnl_perf,
                &bwd_filter_algo));
        
        if(!utils::one_of(bwd_filter_algo, CNNL_CONVOLUTION_BWD_FILTER_ALGO_DIRECT, CNNL_CONVOLUTION_BWD_FILTER_ALGO_GEMM))
            return status::unimplemented;
        
        // cnnl not support winograd convolution
        if(pd->desc()->alg_kind == dnnl_convolution_winograd)
            return status::unimplemented;
        
        // dnnl treat gemm as direct convolution
        utils::downcast<cnnl_convolution_fwd_pd_t *>(pd) -> set_alg_kind(dnnl_convolution_direct);

        return status::success;
    }

    status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto bang_stream = utils::downcast<sycl_bang_stream_t *>(service_stream);
        auto handle = bang_stream->get_cnnl_handle();
        
        scratchpad_size = 0;
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetConvolutionBackwardFilterWorkspaceSize, handle, quantized_src_desc, 
            quantized_dst_desc, descs[io::weights], conv_desc, bwd_filter_algo, &scratchpad_size_conv));
        scratchpad_size = std::max(scratchpad_size_conv, scratchpad_size);
        
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetQuantizeParamWorkspaceSize, handle, descs[y], &scratchpad_size_qA));
        scratchpad_size = std::max(scratchpad_size_qA, scratchpad_size);
        
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetQuantizeParamWorkspaceSize, handle, descs[x], &scratchpad_size_qB));
        scratchpad_size = std::max(scratchpad_size_qB, scratchpad_size);

        if (scratchpad_size > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_cnnl,
                    scratchpad_size, size_t(1));

        return cnnl_convolution_impl_base_t::init_scratchpad(engine, pd);
    }

    void execute(cnnlHandle_t handle, const std::vector<void *> &args) const override {
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

        // quantization
        void *d_q_g_dst, *d_q_src;
        int dst_size = dims[io::y][0]*dims[io::y][1]*dims[io::y][2]*dims[io::y][3];
        int src_size = dims[io::x][0]*dims[io::x][1]*dims[io::x][2]*dims[io::x][3];
        
        cnrtMalloc(&d_q_g_dst, sizeof(int16_t) * dst_size);
        cnrtMalloc(&d_q_src, sizeof(int16_t) * src_size);

        cnrtMemset(d_q_g_dst, 0, sizeof(int16_t) * dst_size);
        cnrtMemset(d_q_src, 0, sizeof(int16_t) * src_size);
        cnrtSyncDevice();

        // Quantize input
        void* scratchpad_qA = scratchpad_size_qA > 0 ? scratchpad : nullptr;
        quantize_array(handle, descs[io::y], y, 16, scratchpad_qA, scratchpad_size_qA, quantized_dst_desc, d_q_g_dst);
        
        // Quantize weight
        void* scratchpad_qB = scratchpad_size_qB > 0 ? scratchpad : nullptr;
        quantize_array(handle, descs[io::x], x, 16, scratchpad_qB, scratchpad_size_qB, quantized_src_desc, d_q_src);

        void* scratchpad_conv = scratchpad_size_conv > 0 ? scratchpad : nullptr;
        CNNL_EXECUTE_FUNC_V(cnnlConvolutionBackwardFilter, handle, &alpha,
                quantized_src_desc, d_q_src, quantized_dst_desc, d_q_g_dst, conv_desc, bwd_filter_algo,
                scratchpad_conv, scratchpad_size_conv, &beta, descs[io::weights], filter);
        if (with_bias) {
            // TODO: check this, especially the "axis" parameter, which equal to the Channl dimention of the feature map(according to the CNNL docs)
            CNNL_EXECUTE_FUNC_V(cnnlBiasAddBackward, handle, descs[io::y], y, dims[io::weights][3], descs[io::bias], bias);
        }
        if (using_transformed_filter()) {
            // TODO: check if this should be support
            undo_transform_filter(handle, filter, weights); // not support
            assert(0);
        }
    }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
