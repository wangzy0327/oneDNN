// MIOpen currently only implements NCHW layout

#ifndef GPU_AMD_MIOPEN_REORDER_HPP
#define GPU_AMD_MIOPEN_REORDER_HPP

#include "common/type_helpers.hpp"
#include "common/engine.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/impl_list_item.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

#include "common/reorder_pd.hpp"
#include "gpu/ocl/cross_engine_reorder.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_reorder_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public reorder_pd_t {
        using reorder_pd_t::reorder_pd_t;
        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_reorder_t);

        // Function to verify data and memory format
        bool valid_data_n_mem_format() const {
            bool ok = utils::one_of(src_md()->data_type, data_type::s8,
                              data_type::f16, data_type::f32)
                    && utils::one_of(dst_md()->data_type, data_type::s8,
                            data_type::f16, data_type::f32);

            // Nvidia only supports blocking for Int8
            if (!utils::one_of(src_md()->data_type, data_type::s8)
                    && src_md()->format_desc.blocking.inner_nblks > 0)
                return false;
            if (!utils::one_of(dst_md()->data_type, data_type::s8)
                    && dst_md()->format_desc.blocking.inner_nblks > 0)
                return false;

            // Nvidia supports blocking only on channel dimension C
            if (dst_md()->format_desc.blocking.inner_nblks > 1
                    || src_md()->format_desc.blocking.inner_nblks > 1)
                return false;
            if (utils::one_of(src_md()->data_type, data_type::s8)
                    && src_md()->format_desc.blocking.inner_nblks == 1) {
                ok = ok && memory_desc_matches_nchw_vect_c(src_md());
            }
            int blks = dst_md()->format_desc.blocking.inner_nblks;
            if (utils::one_of(dst_md()->data_type, data_type::s8)
                    && blks == 1) {
                ok = ok && memory_desc_matches_nchw_vect_c(dst_md());
            }
            return ok;
        }

        bool check_scales_mask() const {
            // cuDNN does not support scaling per dimension. miopen?
            if (attr()->output_scales_.mask_ != 0) { return false; }
            return true;
        }

        status_t init(engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
            bool ok = true && (engine == dst_engine)
                    && (src_engine->kind() == engine_kind::gpu)
                    && valid_data_n_mem_format() && check_scales_mask();
            if (!ok) return status::unimplemented;

            if (has_different_block_size(src_md(), dst_md())) {
                return status::unimplemented;
            } else {
                reorder_.reset(new miopen_reorder_stride_t());
            }

            return reorder_->init(this);
        }
        std::shared_ptr<miopen_reorder_generic_t> reorder_;

    private:
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            auto _pd = new pd_t(attr, src_engine->kind(), src_md,
                    dst_engine->kind(), dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            if (_pd->init(engine, src_engine, dst_engine) != status::success) {
                delete _pd;
                return status::unimplemented;
            }
            _pd->init_scratchpad_md();
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }
        friend dnnl::impl::impl_list_item_t;
    };

    status_t execute(const exec_ctx_t &ctx) const override {
        memory_desc_wrapper wrap(pd()->src_md());
        if (wrap.size() == 0) { return status::success; }

        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());
        return hip_stream->interop_task([&](::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);

            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                        hip_stream->engine());
                auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
                auto handle = hip_stream->get_miopen_handle();

                auto a = sc.memory<uint8_t *>(ih, src_acc)
                        + pd()->reorder_->src_offset_in_bytes();
                auto b = sc.memory<uint8_t *>(ih, dst_acc)
                        + pd()->reorder_->dst_offset_in_bytes();
                pd()->reorder_->execute(handle, a, b);
            });
        });
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct miopen_reorder_generic_t {
public:
    virtual status_t init(const reorder_pd_t *pd) = 0;

    virtual void execute(rocblas_handle handle, void *src, void *dst) const = 0;

    virtual ~miopen_reorder_generic_t() {
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, src_desc_);
        MIOPEN_EXECUTE_FUNC_V(miopenDestroyTensorDescriptor, dst_desc_);
    }

    int dst_offset_in_bytes() { return dst_offset_in_bytes_; }
    int src_offset_in_bytes() { return src_offset_in_bytes_; }

protected:
    miopenDataType_t src_data_type_;
    miopenDataType_t dst_data_type_;
    int ndims_;
    int dims_[DNNL_MAX_NDIMS];
    miopenTensorDescriptor_t src_desc_;
    miopenTensorDescriptor_t dst_desc_;
    float alpha_, beta_;
    int dst_offset_in_bytes_ = 0;
    int src_offset_in_bytes_ = 0;
};

// This structure is used when the memory format includes blocking
// EX TODO

// This structure is used when the memory format does not include blocking
struct miopen_reorder_stride_t : public miopen_reorder_generic_t {
public:
    status_t init(const reorder_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with creating
        // cudnn descriptors
        memory_desc_wrapper wrap(pd->src_md());
        if (wrap.size() == 0) { return status::success; }

        // Validity checks
        assert(pd->dst_md()->ndims == pd->src_md()->ndims);
        dst_offset_in_bytes_ = pd->dst_md()->offset0
                * types::data_type_size(pd->dst_md()->data_type);
        src_offset_in_bytes_ = pd->src_md()->offset0
                * types::data_type_size(pd->src_md()->data_type);
        alpha_ = pd->alpha();
        beta_ = pd->beta();

        convert_dims(pd->dst_md()->dims, dims_, pd->dst_md()->ndims);
        convert_dims(pd->src_md()->format_desc.blocking.strides, src_strides_,
                pd->src_md()->ndims);
        convert_dims(pd->dst_md()->format_desc.blocking.strides, dst_strides_,
                pd->dst_md()->ndims);

        adjust_dim_for_dnn(dims_, pd->dst_md()->ndims, pd->src_md());
        adjust_stride_for_dnn(src_strides_, pd->dst_md()->ndims, pd->src_md());
        adjust_stride_for_dnn(dst_strides_, pd->dst_md()->ndims, pd->dst_md());

        ndims_ = pd->dst_md()->ndims >= 4 ? pd->dst_md()->ndims
                        + pd->dst_md()->format_desc.blocking.inner_nblks
                                          : 4;
        bool vectorized = has_different_block_size(pd->src_md(), pd->dst_md());
        CHECK(convert_data_type(pd->src_md(), &src_data_type_, vectorized));
        CHECK(convert_data_type(pd->dst_md(), &dst_data_type_, vectorized));

        // Create and set source tensor descriptor
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenCreateTensorDescriptor, &src_desc_));
        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetTensorNdDescriptor, src_desc_,
                src_data_type_, ndims_, dims_, src_strides_));
        // Create and set destination tensor descriptor
        CHECK(CUDNN_EXECUTE_FUNC_S(miopenCreateTensorDescriptor, &dst_desc_));
        CHECK(CUDNN_EXECUTE_FUNC_S(miopenSetTensorNdDescriptor, dst_desc_,
                dst_data_type_, ndims_, dims_, dst_strides_));
        return status::success;
    }

    void execute(miopenHandle_t handle, void *src, void *dst) const override {
        // We don't need to specify the format (deducible using the strides)
        // in case of cudnnTransformTensor().
        // For example, this is useful when converting from abcd to bacd
        MIOPEN_EXECUTE_FUNC(miopenTransformTensor, handle, &alpha_, src_desc_,
                src, &beta_, dst_desc_, dst);
    }

private:
    int src_strides_[DNNL_MAX_NDIMS];
    int dst_strides_[DNNL_MAX_NDIMS];

    using miopen_reorder_generic_t::miopen_reorder_generic_t;
};

namespace {
#define INSTANCE(...) \
    impl_list_item_t( \
            impl_list_item_t::reorder_type_deduction_helper_t<__VA_ARGS__>())

// clang-format off
const impl_list_item_t hip_reorder_impl_list[] = {
        INSTANCE(gpu::ocl::cross_engine_reorder_t::pd_t),
        // INSTANCE(miopen_reorder_t::pd_t),
        nullptr,
};
// clang-format on
cnim
} // namespace

const impl_list_item_t *
hip_gpu_engine_impl_list_t::get_reorder_implementation_list(
        const memory_desc_t *, const memory_desc_t *) {
    return hip_reorder_impl_list;
}

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

# endif