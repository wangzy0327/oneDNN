#ifndef GPU_AMD_MIOPEN_POOLING_IMPL_HPP
#define GPU_AMD_MIOPEN_POOLING_IMPL_HPP

#include "miopen/miopen.h"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_pooling_impl_base_t {
    virtual status_t init(const pooling_pd_t *pd) = 0;

    virtual ~miopen_pooling_impl_base_t() {
        for (size_t i = 0; i < NUM_IO; ++i) {
            if (tensor_descs_[i]) {
                MIOPEN_EXECUTE_FUNC_V(
                        miopenDestroyTensorDescriptor, tensor_descs_[i]);
            }
        }
        if (pool_desc_) {
            MIOPEN_EXECUTE_FUNC_V(miopenDestroyPoolingDescriptor, pool_desc_);
        }
    }

    virtual void execute(miopenHandle_t handle, void *x, void *y, void *ws, size_t ws_size) const = 0;

protected:
    status_t init_common(const pooling_pd_t *pd) {
        ndims_ = std::max(4, pd->ndims());
        kernel_ndims_ = ndims_ - 2;

        is_training_ = pd->desc()->prop_kind == prop_kind::forward_training;
        bool is_fwd = pd->is_fwd();
        auto src_md = is_fwd ? pd->src_md() : pd->diff_src_md();
        auto dst_md = is_fwd ? pd->dst_md() : pd->diff_dst_md();

        if (has_zero_dims(src_md->dims, pd->ndims())
                || has_zero_dims(dst_md->dims, pd->ndims())) {
            return status::success;
        }

        if (is_training_) {
            auto src_wrap = memory_desc_wrapper(src_md);
            auto dst_wrap = memory_desc_wrapper(dst_md);
            x_size_bytes_ = src_wrap.size();
            y_size_bytes_ = dst_wrap.size();
        }

        convert_dims(src_md->padded_dims, dims_[src], pd->ndims());
        convert_dims(dst_md->padded_dims, dims_[dst], pd->ndims());

        convert_dims(src_md->format_desc.blocking.strides, strides_[src],
                pd->ndims());
        convert_dims(dst_md->format_desc.blocking.strides, strides_[dst],
                pd->ndims());

        convert_dims(pd->desc()->kernel, kernel_dims_, kernel_ndims_);

        // If 1D pooling
        if (pd->ndims() == 3) {
            // Convert to [n, c, 1, w] since the current format is
            // [n, c, w, 1]
            dims_[src][3] = dims_[src][2];
            dims_[src][2] = 1;

            dims_[dst][3] = dims_[dst][2];
            dims_[dst][2] = 1;

            // Set kernel dimensions to [1, kw]
            kernel_dims_[1] = kernel_dims_[0];
            kernel_dims_[0] = 1;
        }

        if (ndims_ == 4) {
            kernel_padding_[0] = static_cast<int>(pd->padT());
            kernel_padding_[1] = static_cast<int>(pd->padL());

            kernel_strides_[0] = static_cast<int>(pd->KSH());
            kernel_strides_[1] = static_cast<int>(pd->KSW());
        } else {
            kernel_padding_[0] = static_cast<int>(pd->padFront());
            kernel_padding_[1] = static_cast<int>(pd->padT());
            kernel_padding_[2] = static_cast<int>(pd->padL());

            kernel_strides_[0] = static_cast<int>(pd->KSD());
            kernel_strides_[1] = static_cast<int>(pd->KSH());
            kernel_strides_[2] = static_cast<int>(pd->KSW());
        }

        CHECK(convert_data_type(src_md, &data_types_[src]));
        CHECK(convert_data_type(dst_md, &data_types_[dst]));

        CHECK(convert_alg_kind(pd->desc()->alg_kind, &pool_mode_));
        
        // let's just support nchw
        CHECK(verify_format(src_md));
        CHECK(verify_format(dst_md));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[src], data_types_[src], ndims_, dims_[src], strides_[src]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[dst], data_types_[dst], ndims_, dims_[dst], strides_[dst]));

        CHECK(create_and_set_pooling_descriptor(pd));

        return status::success;
    }

    status_t create_and_set_pooling_descriptor(const pooling_pd_t *pd) {
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenCreatePoolingDescriptor, &pool_desc_));

        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetNdPoolingDescriptor, pool_desc_,
                pool_mode_, kernel_ndims_, kernel_dims_,
                kernel_padding_, kernel_strides_));

        return status::success;
    }

    status_t convert_alg_kind(
            alg_kind_t alg_kind, miopenPoolingMode_t *miopen_alg_kind) const {
        switch (alg_kind) {
            case alg_kind::pooling_max:
                *miopen_alg_kind = miopenPoolingMax;
                break;
            case alg_kind::pooling_avg_include_padding:
                *miopen_alg_kind = miopenPoolingAverageInclusive;
                break;
            case alg_kind::pooling_avg_exclude_padding:
                *miopen_alg_kind = miopenPoolingAverage;
                break;
            default: return status::unimplemented;
        }

        return status::success;
    }

    enum io { src = 0, dst, NUM_IO };
    miopenDataType_t data_types_[NUM_IO];
    miopenTensorDescriptor_t tensor_descs_[NUM_IO] = {};
    miopenPoolingDescriptor_t pool_desc_;
    miopenPoolingMode_t pool_mode_ = miopenPoolingMax;
    int dims_[NUM_IO][DNNL_MAX_NDIMS];
    int strides_[NUM_IO][DNNL_MAX_NDIMS];
    int kernel_dims_[DNNL_MAX_NDIMS];
    int kernel_padding_[DNNL_MAX_NDIMS];
    int kernel_strides_[DNNL_MAX_NDIMS];
    const float alpha_ = 1.f, beta_ = 0.f;
    int ndims_, kernel_ndims_;
    bool is_training_ = false;
    std::size_t x_size_bytes_ = 0, y_size_bytes_ = 0;
};

struct miopen_pooling_fwd_impl_t : public miopen_pooling_impl_base_t {
    status_t init(const pooling_pd_t *pd) override {
        return miopen_pooling_impl_base_t::init_common(pd);
    }

    void execute(miopenHandle_t handle, void *x, void *y, void *ws, size_t ws_size) const override {
        MIOPEN_EXECUTE_FUNC(miopenPoolingForward, handle, pool_desc_, 
            &alpha_, tensor_descs_[src], x, 
            &beta_, tensor_descs_[dst], y,
            is_training_, ws, ws_size);
    }
};

struct miopen_pooling_bwd_impl_t : public miopen_pooling_impl_base_t {
    status_t init(const pooling_pd_t *pd) override {
        return miopen_pooling_impl_base_t::init_common(pd);
    }

    void execute(miopenHandle_t handle, void *dx, void *dy, void *ws, size_t /*ws_size*/) const override {
        // miopenPoolingBackward have two not used arguments(input x and output y) orz...
        MIOPEN_EXECUTE_FUNC(miopenPoolingBackward, handle, pool_desc_, 
            &alpha_, tensor_descs_[dst], nullptr, tensor_descs_[dst], dy,
            tensor_descs_[src], nullptr, &beta_, tensor_descs_[src], dx, ws);
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnn

#endif