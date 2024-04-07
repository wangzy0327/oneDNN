#ifndef GPU_CAMBRICON_CNNL_POOLING_IMPL_HPP
#define GPU_CAMBRICON_CNNL_POOLING_IMPL_HPP

#include "cnnl.h"
#include "gpu/cambricon/sycl_bang_utils.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_stream.hpp"
#include "gpu/cambricon/sycl_bang_scoped_context.hpp"
#include <iostream>
#include <chrono>

using namespace std::chrono;

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_pooling_impl_base_t {
    virtual status_t init(engine_t *engine, pooling_pd_t *pd) = 0;

    virtual ~cnnl_pooling_impl_base_t() {
        for (size_t i = 0; i < NUM_IO; ++i) {
            if (tensor_descs_[i]) {
                CNNL_EXECUTE_FUNC_V(
                        cnnlDestroyTensorDescriptor, tensor_descs_[i]);
            }
        }
        if (pool_desc_) {
            CNNL_EXECUTE_FUNC_V(cnnlDestroyPoolingDescriptor, pool_desc_);
        }
    }

    virtual void execute(cnnlHandle_t handle, void *x, void *y, void *scratchpad) const = 0;
    
    bool with_scratchpad() const { return scratchpad_size > 0; }

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
            kernel_padding_[2] = static_cast<int>(pd->padB());
            kernel_padding_[3] = static_cast<int>(pd->padR());

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
        
        // cnnl pooling just support nhwc and nchw 
        CHECK(get_format(src_md, src_format));
        CHECK(get_format(dst_md, dst_format));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[src], src_format,  data_types_[src], ndims_, dims_[src]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[dst], dst_format,  data_types_[dst], ndims_, dims_[dst]));

        CHECK(create_and_set_pooling_descriptor(pd));

        return status::success;
    }

    status_t create_and_set_pooling_descriptor(const pooling_pd_t *pd) {
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlCreatePoolingDescriptor, &pool_desc_));

        CHECK(CNNL_EXECUTE_FUNC_S(cnnlSetPoolingNdDescriptor, pool_desc_, 
                pool_mode_, CNNL_NOT_PROPAGATE_NAN, kernel_ndims_, kernel_dims_,
                kernel_padding_, kernel_strides_));

        return status::success;
    }

    status_t convert_alg_kind(
            alg_kind_t alg_kind, cnnlPoolingMode_t *cnnl_alg_kind) const {
        switch (alg_kind) {
            case alg_kind::pooling_max:
                *cnnl_alg_kind = CNNL_POOLING_MAX;
                break;
            case alg_kind::pooling_avg_include_padding:
                *cnnl_alg_kind = CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
                break;
            case alg_kind::pooling_avg_exclude_padding:
                *cnnl_alg_kind = CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
                break;
            default: return status::unimplemented;
        }

        return status::success;
    }

    enum io { src = 0, dst, NUM_IO };
    cnnlDataType_t data_types_[NUM_IO];
    cnnlTensorDescriptor_t tensor_descs_[NUM_IO] = {};
    cnnlPoolingDescriptor_t pool_desc_;
    cnnlPoolingMode_t pool_mode_ = CNNL_POOLING_MAX;
    cnnlTensorLayout_t src_format, dst_format;
    size_t scratchpad_size = 0;
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

struct cnnl_pooling_fwd_impl_t : public cnnl_pooling_impl_base_t {
    status_t init(engine_t *engine, pooling_pd_t *pd) override {
        CHECK(init_common(pd));
        
        auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto bang_stream = utils::downcast<sycl_bang_stream_t *>(service_stream);
        auto handle = bang_stream->get_cnnl_handle();

        // allocate workspace, TODO: distinguish the outH and outW parameter by datalayout
        assert(dst_format == CNNL_LAYOUT_NHWC);
        CNNL_EXECUTE_FUNC(cnnlGetPoolingWorkspaceSize, handle, pool_mode_, dims_[io::dst][2], dims_[io::dst][1], &scratchpad_size);
        
        if (scratchpad_size > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_pool_cnnl,
                    scratchpad_size, size_t(1));
        return status::success;
    }

    void execute(cnnlHandle_t handle, void *x, void *y, void *scratchpad) const override {
        CNNL_EXECUTE_FUNC(cnnlPoolingForward, handle, pool_desc_, 
            &alpha_, tensor_descs_[src], x, 
            &beta_, tensor_descs_[dst], y, scratchpad, scratchpad_size);
    }
};

struct cnnl_pooling_bwd_impl_t : public cnnl_pooling_impl_base_t {
    status_t init(engine_t *engine, pooling_pd_t *pd) override {
        return cnnl_pooling_impl_base_t::init_common(pd);
    }

    void execute(cnnlHandle_t handle, void *dx, void *dy, void *scratchpad) const override {
        // cnnlPoolingBackward have two not used arguments(input x and output y) orz...
        cnrtSyncDevice();
        CNNL_EXECUTE_FUNC(cnnlPoolingBackward, handle, pool_desc_, 
            &alpha_, tensor_descs_[dst], nullptr, tensor_descs_[dst], dy,
            tensor_descs_[src], nullptr, &beta_, tensor_descs_[src], dx);
    }
};

} // namespace bang
} // namespace gpu
} // namespace impl
} // namespace dnn

#endif