#ifndef GPU_CAMBRICON_CNNL_CONVOLUTION_HPP
#define GPU_CAMBRICON_CNNL_CONVOLUTION_HPP

#include "cnnl.h"

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc.hpp"
#include "gpu/cambricon/cnnl_convolution_impl.hpp"
#include "gpu/cambricon/cnnl_convolution_pd.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_convolution_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public cnnl_convolution_fwd_pd_t {
        using cnnl_convolution_fwd_pd_t::cnnl_convolution_fwd_pd_t;
        pd_t(const pd_t &other)
            : cnnl_convolution_fwd_pd_t(other)
            , impl_(other.impl_)
            , dst_md_temp_(other.dst_md_temp_) {}

        DECLARE_COMMON_PD_T("bang:cnnl:any", cnnl_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::oscale
                    | primitive_attr_t::skip_mask_t::post_ops;

            bool ok = utils::one_of(desc()->prop_kind, prop_kind::forward_training, prop_kind::forward_inference);
            ok = ok && attr()->has_default_values(attr_skip_mask);
            ok = ok && (utils::everyone_is(f32, src_md_.data_type,
                                weights_md_.data_type, dst_md_.data_type)
                            || utils::everyone_is(f16, src_md_.data_type,
                                    weights_md_.data_type, dst_md_.data_type)); // only f32 and f16 implemented, for cnnl
            ok = ok && this->set_default_formats();
            ok = ok && memory_format_ok(&src_md_);  // check blocked format?
            ok = ok && memory_format_ok(&weights_md_);
            ok = ok && memory_format_ok(&dst_md_);
            if (with_bias()) ok = ok && memory_format_ok(&bias_md_);
            if (!ok) return status::unimplemented;

            if (check_for_zero_dims()) return status::success;

            // TODO: check if temp_dst is necessary
            // const bool use_temp_dst = attr()->post_ops_.len() > 0;
            const bool use_temp_dst = false;
            if (use_temp_dst) {
                dst_md_temp_ = dst_md_;
                if (dst_md_.data_type == s8) { dst_md_temp_.data_type = f32; }
            }

            impl_.reset(new cnnl_convolution_impl_fwd_t());
            return impl_->init(engine, this, use_temp_dst);
        }
        bool with_scratchpad() const { return impl_->with_scratchpad(); }
        std::shared_ptr<cnnl_convolution_impl_base_t> impl_;
        memory_desc_t dst_md_temp_;

        bool use_temp_dst() const {
            if (impl_.get()) return impl_->use_temp_dst();
            return false;
        }

    private:
        // TODO: cambricon localization
        bool set_default_formats() {
            using namespace format_tag;
            if (src_md_.data_type == dnnl_s8) {
                return false;   // not support
            } else {
                auto dat_tag = utils::pick(ndims() - 4, nhwc, ndhwc);
                auto wei_tag = with_groups()
                        ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                        : utils::pick(ndims() - 4, ohwi, odhwi);
                return set_default_formats_common(dat_tag, wei_tag, dat_tag);
            }
        }
    };

    status_t init_temp_dst(engine_t *engine) {
        auto sycl_engine = utils::downcast<sycl_bang_engine_t *>(engine);
        memory_storage_t *scratch_ptr = nullptr;
        auto wrap = memory_desc_wrapper(pd()->dst_md_temp_);
        CHECK(sycl_engine->create_memory_storage(
                &scratch_ptr, memory_flags_t::alloc, wrap.size(), nullptr));
        scratch_storage.reset(scratch_ptr);

        CHECK(sycl_engine->create_memory_storage(
                &scratch_ptr, memory_flags_t::alloc, wrap.size(), nullptr));
        scratch_storage_2.reset(scratch_ptr);

        return status::success;
    }

    virtual status_t init(engine_t *engine) override {
        const auto impl = pd()->impl_.get();
        if (impl && impl->use_temp_dst()) { init_temp_dst(engine); }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->check_for_zero_dims()) { return status::success; }
        execute_convolution(ctx, pd()->with_bias(), pd()->with_scratchpad());
        return status::success;
    }
    status_t execute_convolution(const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const;

private:
    ::sycl::buffer<uint8_t, 1> &buffer(memory_storage_t *mem_storage) const {
        return utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                mem_storage)->buffer();
    }
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<memory_storage_t> scratch_storage;
    std::shared_ptr<memory_storage_t> scratch_storage_2;
};

struct cnnl_convolution_bwd_data_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public cnnl_convolution_bwd_data_pd_t {
        using cnnl_convolution_bwd_data_pd_t::cnnl_convolution_bwd_data_pd_t;

        DECLARE_COMMON_PD_T("bang:cnnl:any", cnnl_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            bool ok = desc()->prop_kind == prop_kind::backward_data;
            ok = ok && this->set_default_formats();
            ok = ok && (utils::everyone_is(f32, diff_src_md_.data_type,
                                weights_md_.data_type, diff_dst_md_.data_type)
                            || utils::everyone_is(f16, diff_src_md_.data_type,
                                    weights_md_.data_type,
                                    diff_dst_md_.data_type));
            ok = ok && memory_format_ok(&diff_src_md_);
            ok = ok && memory_format_ok(&weights_md_);
            ok = ok && memory_format_ok(&diff_dst_md_);
            if (with_bias()) {
                ok = ok && memory_format_ok(&bias_md_);
                ok = ok && bias_md_.data_type == diff_dst_md_.data_type;
            }
            if (!ok) return status::unimplemented;

            if (check_for_zero_dims()) return status::success;

            impl_.reset(new cnnl_convolution_impl_bwd_data_t());
            return impl_->init(engine, this);
        }

        std::shared_ptr<cnnl_convolution_impl_base_t> impl_;

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 4, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 4, ohwi, odhwi);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
        bool with_scratchpad() const { return impl_->with_scratchpad(); }
        bool support_bias() const override { return true; }
    };

    ~cnnl_convolution_bwd_data_t() {}
    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->check_for_zero_dims()) { return status::success; }
        return execute_convolution(ctx, pd()->with_bias(), pd()->with_scratchpad());
    }
    status_t execute_convolution(const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct cnnl_convolution_bwd_weights_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public cnnl_convolution_bwd_weights_pd_t {
        using cnnl_convolution_bwd_weights_pd_t::
                cnnl_convolution_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T("bang:cnnl:any", cnnl_convolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            bool ok = desc()->prop_kind == prop_kind::backward_weights;
            ok = ok && this->set_default_formats();
            ok = ok && (utils::everyone_is(f32, src_md_.data_type,
                                diff_weights_md_.data_type,
                                diff_dst_md_.data_type)
                            || utils::everyone_is(f16, src_md_.data_type,
                                    diff_weights_md_.data_type,
                                    diff_dst_md_.data_type));
            ok = ok && memory_format_ok(&src_md_);
            ok = ok && memory_format_ok(&diff_weights_md_);
            ok = ok && memory_format_ok(&diff_dst_md_);
            if (with_bias()) {
                ok = ok && memory_format_ok(&diff_bias_md_);
                ok = ok && diff_bias_md_.data_type == diff_dst_md_.data_type;
            }
            if (!ok) return status::unimplemented;

            impl_.reset(new cnnl_convolution_impl_bwd_weights_t());
            if (check_for_zero_dims()) { return impl_->init_zero_dims(this); };

            return impl_->init(engine, this);
        }

        std::shared_ptr<cnnl_convolution_impl_base_t> impl_;

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 4, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 4, ohwi, odhwi);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
        bool with_scratchpad() const { return impl_->with_scratchpad(); }
    };

    ~cnnl_convolution_bwd_weights_t() {}
    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->check_for_zero_dims()) { return execute_zero_dims(ctx); }
        return execute_convolution(ctx, pd()->with_bias(), pd()->with_scratchpad());
    }
    status_t execute_convolution(const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const;
    status_t execute_zero_dims(const exec_ctx_t &ctx) const;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
