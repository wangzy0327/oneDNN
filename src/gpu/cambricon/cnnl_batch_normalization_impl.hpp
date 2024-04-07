/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2020 Codeplay Software Limited
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_CAMBRICON_CNNL_BATCH_NORMALIZATION_IMPL_HPP
#define GPU_CAMBRICON_CNNL_BATCH_NORMALIZATION_IMPL_HPP

#include <cnnl.h>

#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct bnorm_args_t {
public:
    bnorm_args_t(void *x, void *mean, void *var, void *scale, void *bias)
        : x_(x), mean_(mean), var_(var), scale_(scale), bias_(bias) {}

    void *x_, *mean_, *var_, *scale_, *bias_;
};

struct bnorm_fwd_args_t : public bnorm_args_t {
    bnorm_fwd_args_t(void *x, void *y, void *weight, void *mean, void *var, void *scale,
            void *bias, void *y_prime, void *save_mean, void *save_var)
        : bnorm_args_t::bnorm_args_t(x, mean, var, scale, bias)
        , weight_(weight)
        , y_(y)
        , y_prime_(y_prime)
        , save_mean_(save_mean)
        , save_var_(save_var) {}

    void *weight_, *y_, *y_prime_, *save_mean_, *save_var_;
};

struct bnorm_bwd_args_t : public bnorm_args_t {
    bnorm_bwd_args_t(void *x, void *dx, void *dy, void *weight, void *mean, void *var,
            void *scale, void *bias, void *diff_scale, void *diff_bias,
            void *wkspace, void *relu_dx)
        : bnorm_args_t(x, mean, var, scale, bias)
        , weight_(weight)
        , dx_(dx)
        , dy_(dy)
        , diff_scale_(diff_scale)
        , diff_bias_(diff_bias)
        , wkspace_(wkspace)
        , relu_dx_(relu_dx) {}

    void *dx_, *dy_, *weight_, *diff_scale_, *diff_bias_, *wkspace_, *relu_dx_;
};

struct cnnl_batch_normalization_impl_base_t {
    virtual ~cnnl_batch_normalization_impl_base_t() {
        for (size_t i = 0; i < NUM_IO; ++i) {
            if (tensor_descs_[i]) {
                CNNL_EXECUTE_FUNC_V(
                        cnnlDestroyTensorDescriptor, tensor_descs_[i]);
            }
        }

        if ((fuse_norm_relu_ || with_relu_postop_) && act_desc_) {
            CNNL_EXECUTE_FUNC_V(cnnlDestroyActivationDescriptor, act_desc_);
        }
    }

    virtual status_t init(batch_normalization_pd_t *pd) = 0;

    virtual void execute(
            cnnlHandle_t handle, std::shared_ptr<bnorm_args_t> args) const = 0;

    bool is_bwd_d() const { return is_bwd_data_; }
    bool is_training() const { return is_training_; }
    bool fuse_norm_relu() const { return fuse_norm_relu_; }
    std::size_t dt_size() const { return dt_size_; }
    std::size_t mean_var_size_bytes() { return mean_var_size_bytes_; }
    uint8_t default_mean_var() const { return 0; }

    
    status_t convert_tag(const memory_desc_t *md, format_tag_t &tag) const {
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

    status_t convert_format(format_tag_t &tag, cnnlTensorLayout_t &format){

        format = (tag == dnnl_nc) ? CNNL_LAYOUT_NC : (tag == dnnl_nwc) ? CNNL_LAYOUT_NLC : (tag == dnnl_nhwc) ? CNNL_LAYOUT_NHWC : (tag == dnnl_nchw) ? CNNL_LAYOUT_NCHW : CNNL_LAYOUT_NDHWC;
        return status::success;

}
    int C() const { return nchannels_; }

protected:
    status_t init_common(batch_normalization_pd_t *pd) {
        ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();
        if (ndims_ > 5) { return status::invalid_arguments; }

        memory_desc_wrapper wrap(pd->src_md());
        fuse_norm_relu_ = pd->fuse_norm_relu();
        is_training_ = pd->is_training();
        with_global_stats_ = pd->use_global_stats();
        is_bwd_data_ = pd->desc()->prop_kind == prop_kind::backward_data;
        dt_size_ = types::data_type_size(wrap.data_type());
        nchannels_ = pd->C();
        mean_var_size_bytes_ = nchannels_ * dt_size_;
        eps_ = pd->desc()->batch_norm_epsilon;
        y_prime_size_ = wrap.nelems() * dt_size_;
        with_relu_postop_ = pd->with_relu_post_op();

        auto n = static_cast<float>(pd->MB() * pd->D() * pd->H() * pd->W());
        var_scaling_factor_ = (n - 1.f) / n;

        convert_dims(pd->src_md()->padded_dims, dims_[src], pd->ndims());

        CHECK(convert_data_type(pd->src_md(), &data_types_[src]));
        CHECK(convert_tag(pd->src_md(), tag));
        CHECK(convert_format(tag,format));
        
        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[src], format,
                data_types_[src], ndims_, dims_[src]));

        if (fuse_norm_relu_ || with_relu_postop_) {
            CHECK(create_and_set_activation_desc());
        }

        return status::success;
    }

    virtual status_t create_and_set_activation_desc() {
        CHECK(CNNL_EXECUTE_FUNC_S(
                cnnlCreateActivationDescriptor, &act_desc_));

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetActivationDescriptor_v3, act_desc_,
                CNNL_ACTIVATION_RELU,  CNNL_ACTIVATION_FAST , CNNL_PROPAGATE_NAN, relu_coef_, 0));

        return status::success;
    }

    enum io { src = 0, dst, wbmv, NUM_IO };
    cnnlDataType_t data_types_[NUM_IO];
    cnnlTensorDescriptor_t tensor_descs_[NUM_IO] = {};
    cnnlActivationDescriptor_t act_desc_=NULL;
    cnnlBatchNormMode_t mode_ = CNNL_BATCHNORM_SPATIAL;
    format_tag_t tag;
    cnnlTensorLayout_t format;
    int dims_[NUM_IO][DNNL_MAX_NDIMS];
    int strides_[NUM_IO][DNNL_MAX_NDIMS];
    int ndims_, nchannels_;
    float alpha_ = 1.f, beta = 0.f;
    double relu_coef_ = 0.0;
    double factor_ = 1.0;
    double eps_ = 0.0000001;
    float var_scaling_factor_ = 0.f;
    bool fuse_norm_relu_ = false;
    bool with_relu_postop_ = false;
    bool with_global_stats_ = false;
    bool is_training_ = false;
    bool is_bwd_data_ = false;
    std::size_t y_prime_size_;
    std::size_t dt_size_, mean_var_size_bytes_;
};

struct cnnl_batch_normalization_fwd_impl_t
    : public cnnl_batch_normalization_impl_base_t {
    using cnnl_batch_normalization_impl_base_t::
            cnnl_batch_normalization_impl_base_t;

    status_t init(batch_normalization_pd_t *pd) override {
        init_common(pd);

        convert_dims(pd->dst_md()->padded_dims, dims_[dst], pd->ndims());

        dims_[wbmv][0]=dims_[src][pd->ndims()-1];   // for nhwc
        
        CHECK(convert_data_type(pd->dst_md(), &data_types_[dst]));
        
        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[wbmv], CNNL_LAYOUT_ARRAY, data_types_[src], 1, dims_[wbmv]));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[dst], format, 
                data_types_[dst], ndims_, dims_[dst]));

        return status::success;
    }

    void execute(cnnlHandle_t handle, std::shared_ptr<bnorm_args_t> args) const override {
        auto fwd_args = static_cast<bnorm_fwd_args_t *>(args.get());

        CNNL_EXECUTE_FUNC(cnnlBatchNormForwardTrainingV2, handle, act_desc_, mode_,
                CNNL_BATCHNORM_OPS_BN_ACTIVATION, &alpha_, &beta, tensor_descs_[src], fwd_args->x_, 
                NULL, NULL,
                tensor_descs_[wbmv], fwd_args->weight_, fwd_args->bias_, fwd_args->mean_,
                fwd_args->var_, eps_, 0,  tensor_descs_[dst], fwd_args->y_,
                fwd_args->save_mean_, fwd_args->save_var_, NULL, 0, NULL, 0);

        //if (is_training_) { to_population_variance(handle, fwd_args->var_); }

        if (fuse_norm_relu_ || with_relu_postop_) { do_relu(handle, fwd_args); }
    }

protected:
    void do_relu(cnnlHandle_t handle, bnorm_fwd_args_t *fwd_args) const {
        if (is_training_ && fuse_norm_relu_) {
            // Copy the result to the workspace
            CNNL_EXECUTE_FUNC(cnnlTransform, handle, &alpha_,
                    tensor_descs_[dst], fwd_args->y_, &beta, fwd_args->y_prime_);
        }

        CNNL_EXECUTE_FUNC(cnnlActivationForward, handle, act_desc_, &alpha_,
                tensor_descs_[dst], fwd_args->y_, &beta, tensor_descs_[dst],
                fwd_args->y_);
    }
};

struct cnnl_batch_normalization_fwd_stats_impl_t
    : public cnnl_batch_normalization_fwd_impl_t {

    status_t init(batch_normalization_pd_t *pd) override {
        return cnnl_batch_normalization_fwd_impl_t::init(pd);
    }

    void execute(cnnlHandle_t handle,
            std::shared_ptr<bnorm_args_t> args) const override {
        // printf("hi...\n");
        auto fwd_args = static_cast<bnorm_fwd_args_t *>(args.get());
        CNNL_EXECUTE_FUNC(cnnlBatchNormForwardInference, handle,
                &alpha_, &beta, tensor_descs_[src], fwd_args->x_,
                tensor_descs_[wbmv], fwd_args->weight_, fwd_args->bias_, fwd_args->mean_,
                fwd_args->var_, eps_, tensor_descs_[dst], fwd_args->y_);

        if (fuse_norm_relu_ || with_relu_postop_) { do_relu(handle, fwd_args); }
    }
};

struct cnnl_batch_normalization_bwd_impl_t
    : public cnnl_batch_normalization_impl_base_t {

    status_t init(batch_normalization_pd_t *pd) override {
        init_common(pd);

        convert_dims(pd->diff_src_md()->padded_dims, diff_dims_[diff_src],
                pd->ndims());
        convert_dims(pd->diff_dst_md()->padded_dims, diff_dims_[diff_dst],
                pd->ndims());

        dims_[wbmv][0]=dims_[src][pd->ndims()-1];

        convert_dims(pd->diff_src_md()->format_desc.blocking.strides,
                strides_[diff_src], pd->ndims());
        convert_dims(pd->diff_dst_md()->format_desc.blocking.strides,
                strides_[diff_dst], pd->ndims());

        CHECK(convert_data_type(
                pd->diff_src_md(), &diff_data_types_[diff_src]));
        CHECK(convert_data_type(
                pd->diff_dst_md(), &diff_data_types_[diff_dst]));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[wbmv], CNNL_LAYOUT_ARRAY, data_types_[src], 1, dims_[wbmv])); 

        CHECK(create_and_set_tensor_descriptor(&diff_tensor_descs_[diff_src], format, 
                data_types_[diff_src], ndims_, diff_dims_[diff_src]));
        CHECK(create_and_set_tensor_descriptor(&diff_tensor_descs_[diff_dst], format,
                data_types_[diff_dst], ndims_, diff_dims_[diff_dst]));

        return status::success;
    }

    void execute(cnnlHandle_t handle,
            std::shared_ptr<bnorm_args_t> args) const override {
        auto bwd_args = static_cast<bnorm_bwd_args_t *>(args.get());
        

        CNNL_EXECUTE_FUNC(cnnlBatchNormBackward, handle, 
                &a_data_diff_, &b_data_diff_, &a_param_diff_, &b_param_diff_,
                tensor_descs_[src], bwd_args->x_, diff_tensor_descs_[diff_dst],
                bwd_args->dy_, tensor_descs_[wbmv], bwd_args->weight_, bwd_args->mean_, bwd_args->var_,  eps_,
                diff_tensor_descs_[diff_src], bwd_args->dx_, bwd_args->diff_scale_, bwd_args->diff_bias_
                );
    }

    ~cnnl_batch_normalization_bwd_impl_t() {
        for (size_t i = 0; i < NUM_DIFF; i++) {
            if (diff_tensor_descs_[i]) {
                CNNL_EXECUTE_FUNC_V(
                        cnnlDestroyTensorDescriptor, diff_tensor_descs_[i]);
            }
        }
    }

protected:
    const float a_data_diff_ = 1.f, b_data_diff_ = 0.f;
    const float a_param_diff_ = 1.f, b_param_diff_ = 0.f;

    enum diff_tensors { diff_src = 0, diff_dst, NUM_DIFF };
    int diff_dims_[NUM_DIFF][DNNL_MAX_NDIMS];
    cnnlTensorDescriptor_t diff_tensor_descs_[NUM_DIFF] = {};
    cnnlDataType_t diff_data_types_[NUM_DIFF];
};

struct cnnl_batch_normalization_bwd_relu_impl_t
    : public cnnl_batch_normalization_bwd_impl_t {

    status_t init(batch_normalization_pd_t *pd) override {
        pd->scratchpad_registry().registrar().book(
                memory_tracking::names::key_none,
                memory_desc_wrapper(pd->diff_dst_md()).size(), size_t(1));

        return cnnl_batch_normalization_bwd_impl_t::init(pd);
    }

    void execute(cnnlHandle_t handle,
            std::shared_ptr<bnorm_args_t> args) const override {
        auto bwd_args = static_cast<bnorm_bwd_args_t *>(args.get());

        CNNL_EXECUTE_FUNC(cnnlActivationBackward, handle, act_desc_, &alpha_, NULL,  NULL,
                diff_tensor_descs_[dst], bwd_args->dy_,  tensor_descs_[src], bwd_args->x_, &beta, diff_tensor_descs_[dst],
                bwd_args->relu_dx_);

        CNNL_EXECUTE_FUNC(cnnlBatchNormBackward, handle, 
                &a_data_diff_, &b_data_diff_, &a_param_diff_, &b_param_diff_,
                tensor_descs_[src], bwd_args->x_, diff_tensor_descs_[diff_dst],
                bwd_args->dy_, tensor_descs_[wbmv],bwd_args->weight_, bwd_args->mean_, bwd_args->var_,  eps_,
                diff_tensor_descs_[diff_src], bwd_args->dx_, bwd_args->diff_scale_, bwd_args->diff_bias_
                );
    }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
