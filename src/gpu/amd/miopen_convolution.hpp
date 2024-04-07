#ifndef GPU_AMD_MIOPEN_CONVOLUTION_HPP
#define GPU_AMD_MIOPEN_CONVOLUTION_HPP

#include <miopen/miopen.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc.hpp"
#include "gpu/amd/miopen_convolution_impl.hpp"
#include "gpu/amd/miopen_convolution_pd.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_convolution_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public miopen_convolution_fwd_pd_t {
        using miopen_convolution_fwd_pd_t::miopen_convolution_fwd_pd_t;
        pd_t(const pd_t &other)
            : miopen_convolution_fwd_pd_t(other)
            , impl_(other.impl_)
            , dst_md_temp_(other.dst_md_temp_) {}

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::oscale
                    | primitive_attr_t::skip_mask_t::post_ops;

            bool ok = utils::one_of(desc()->prop_kind,
                    prop_kind::forward_training, prop_kind::forward_inference);
            ok = ok && attr()->has_default_values(attr_skip_mask);
            ok = ok && post_ops_ok(attr());
            ok = ok
                    && (utils::everyone_is(f32, src_md_.data_type,
                                weights_md_.data_type, dst_md_.data_type)
                            || utils::everyone_is(f16, src_md_.data_type,
                                    weights_md_.data_type, dst_md_.data_type)
                            || (utils::everyone_is(s8, src_md_.data_type,
                                        weights_md_.data_type)
                                    && utils::one_of(
                                            dst_md_.data_type, f32, s8)));
            ok = ok && this->set_default_formats();
            ok = ok
                    && IMPLICATION(
                            desc()->alg_kind == dnnl_convolution_winograd,
                            ndims() < 5 && src_md_.data_type != s8);
            ok = ok
                    && IMPLICATION(!attr()->output_scales_.has_default_values(),
                            src_md_.data_type == s8
                                    && attr()->output_scales_.mask_ == 0);
            // ok = ok
            //         && IMPLICATION(src_md_.data_type == s8, check_s8_configuration());

            ok = ok && memory_format_ok(&src_md_);
            ok = ok && memory_format_ok(&weights_md_);
            ok = ok && memory_format_ok(&dst_md_);
            if (with_bias()) ok = ok && memory_format_ok(&bias_md_);
            if (!ok) return status::unimplemented;

            if (check_for_zero_dims()) return status::success;

            const bool use_temp_dst = attr()->post_ops_.len() > 0;
            if (use_temp_dst) {
                dst_md_temp_ = dst_md_;
                if (dst_md_.data_type == s8) { dst_md_temp_.data_type = f32; }
            }

            impl_.reset(new miopen_convolution_impl_fwd_t());
            return impl_->init(engine, this, use_temp_dst);
        }
        bool with_scratchpad() const { return impl_->with_scratchpad(); }
        std::shared_ptr<miopen_convolution_impl_base_t> impl_;
        memory_desc_t dst_md_temp_;

        bool use_temp_dst() const {
            if (impl_.get()) return impl_->use_temp_dst();
            return false;
        }

    private:
        bool set_default_formats() {
            using namespace format_tag;
            if (src_md_.data_type == dnnl_s8) {
                auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
                auto wei_tag = with_groups()
                        ? utils::pick(ndims() - 3, gowi, gohwi, godhwi)
                        : utils::pick(ndims() - 3, owi, ohwi, odhwi);
                return set_default_formats_common(dat_tag, wei_tag, dat_tag);
            } else {
                auto dat_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
                auto wei_tag = with_groups()
                        ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                        : utils::pick(ndims() - 3, oiw, oihw, oidhw);
                return set_default_formats_common(dat_tag, wei_tag, dat_tag);
            }
        }

        bool post_ops_ok(const primitive_attr_t *attr) const {
            const auto &p = attr->post_ops_;
            auto is_eltwise
                    = [&](int idx) { return p.entry_[idx].is_eltwise(false); };
            auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(false); };

            switch (p.len()) {
                case 0: return true; // no post_ops
                case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
                case 2:
                    if (src_md_.data_type == dnnl_s8 && is_eltwise(0)
                            && is_sum(1))
                        return true;
                    return (is_sum(0) && is_eltwise(1));
                default: return false;
            }

            return false;
        }
    };

    status_t init_temp_dst(engine_t *engine) {
        auto sycl_engine = utils::downcast<sycl_hip_engine_t *>(engine);
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
    
    status_t execute_convolution(const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const{
        amd::sycl_hip_stream_t *hip_stream
                = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([&](::sycl::handler &cgh) {
            using scratch_acc_t = ::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::read_write>;
            auto x_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto weights_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto y_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            std::shared_ptr<
                    ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::read>>
                    bias_acc;
            std::shared_ptr<scratch_acc_t> scratch_acc;
            std::shared_ptr<scratch_acc_t> filter_scratch_acc;
            std::shared_ptr<scratch_acc_t> temp_dst_acc;
            std::shared_ptr<scratch_acc_t> temp_reorder_acc;

            const bool use_temp_dst = pd()->use_temp_dst();

            if (with_scratchpad) {
                scratch_acc = std::make_shared<scratch_acc_t>(
                        utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                                ctx.get_scratchpad_grantor()
                                        .get_memory_storage(memory_tracking::names::
                                                        key_conv_cudnn_algo)
                                        .get())
                                ->buffer()
                                .get_access<::sycl::access::mode::read_write>(cgh));
            }
            if (with_bias) {
                bias_acc = std::make_shared<
                        ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::read>>(
                        CTX_IN_ACCESSOR(DNNL_ARG_BIAS));
            }
            if (pd()->impl_->using_transformed_filter()) {
                filter_scratch_acc
                        = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                                memory_tracking::names::key_conv_cudnn_filter));
            }

            if (use_temp_dst) {
                temp_dst_acc = std::make_shared<scratch_acc_t>(
                        buffer(scratch_storage.get())
                                .get_access<::sycl::access::mode::read_write>(cgh));
                temp_reorder_acc = std::make_shared<scratch_acc_t>(
                        buffer(scratch_storage_2.get())
                                .get_access<::sycl::access::mode::read_write>(cgh));
            }

            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                        hip_stream->engine());
                auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
                auto handle = hip_stream->get_miopen_handle();

                std::vector<void *> args;
                args.push_back(sc.memory<void *>(ih, x_acc));
                args.push_back(sc.memory<void *>(ih, weights_acc));
                args.push_back(sc.memory<void *>(ih, y_acc));
                args.push_back(
                        with_bias ? sc.memory<void *>(ih, *bias_acc) : nullptr);
                args.push_back(with_scratchpad ? sc.memory<void *>(ih, *scratch_acc)
                                            : nullptr);
                args.push_back(pd()->impl_->using_transformed_filter()
                                ? sc.memory<void *>(ih, *filter_scratch_acc)
                                : nullptr);
                args.push_back(use_temp_dst ? sc.memory<void *>(ih, *temp_dst_acc)
                                            : nullptr);
                args.push_back(use_temp_dst
                                ? sc.memory<void *>(ih, *temp_reorder_acc)
                                : nullptr);
                pd()->impl_->execute(handle, args);
            });
        });
    }

private:
    ::sycl::buffer<uint8_t, 1> &buffer(memory_storage_t *mem_storage) const {
        return utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                mem_storage)
                ->buffer();
    }
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<memory_storage_t> scratch_storage;
    std::shared_ptr<memory_storage_t> scratch_storage_2;
};

struct miopen_convolution_bwd_data_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public miopen_convolution_bwd_data_pd_t {
        using miopen_convolution_bwd_data_pd_t::miopen_convolution_bwd_data_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            bool ok = desc()->prop_kind == prop_kind::backward_data;
            ok = ok && this->set_default_formats();
            ok = ok
                    && (utils::everyone_is(f32, diff_src_md_.data_type,
                                weights_md_.data_type, diff_dst_md_.data_type)
                            || utils::everyone_is(f16, diff_src_md_.data_type,
                                    weights_md_.data_type,
                                    diff_dst_md_.data_type));

            ok = ok
                    && IMPLICATION(
                            desc()->alg_kind == dnnl_convolution_winograd,
                            ndims() < 5);
            ok = ok && memory_format_ok(&diff_src_md_);
            ok = ok && memory_format_ok(&weights_md_);
            ok = ok && memory_format_ok(&diff_dst_md_);
            if (with_bias()) {
                ok = ok && memory_format_ok(&bias_md_);
                ok = ok && bias_md_.data_type == diff_dst_md_.data_type;
            }
            if (!ok) return status::unimplemented;

            if (check_for_zero_dims()) return status::success;

            impl_.reset(new miopen_convolution_impl_bwd_data_t());
            return impl_->init(engine, this);
        }

        std::shared_ptr<miopen_convolution_impl_base_t> impl_;

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
        bool with_scratchpad() const { return impl_->with_scratchpad(); }
        bool support_bias() const override { return true; }
    };

    ~miopen_convolution_bwd_data_t() {}
    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->check_for_zero_dims()) { return status::success; }
        return execute_convolution(
                ctx, pd()->with_bias(), pd()->with_scratchpad());
    }
    status_t execute_convolution(const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const{
        amd::sycl_hip_stream_t *hip_stream
            = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([&](::sycl::handler &cgh) {
            using scratch_acc_t = ::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::read_write>;
            auto x_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
            auto weights_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto y_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
            std::shared_ptr<
                    ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::read>>
                    bias_acc;
            std::shared_ptr<scratch_acc_t> scratch_acc;
            std::shared_ptr<scratch_acc_t> filter_scratch_acc;
            if (with_scratchpad) {
                scratch_acc = std::make_shared<scratch_acc_t>(
                        utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                                ctx.get_scratchpad_grantor()
                                        .get_memory_storage(memory_tracking::names::
                                                        key_conv_cudnn_algo)
                                        .get())
                                ->buffer()
                                .get_access<::sycl::access::mode::read_write>(cgh));
            }
            if (with_bias) {
                bias_acc = std::make_shared<
                        ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::read>>(
                        CTX_IN_ACCESSOR(DNNL_ARG_BIAS));
            }
            if (pd()->impl_->using_transformed_filter()) {
                filter_scratch_acc
                        = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                                memory_tracking::names::key_conv_cudnn_filter));
            }
            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                        hip_stream->engine());
                auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
                auto handle = hip_stream->get_miopen_handle();

                std::vector<void *> args;
                args.push_back(sc.memory<void *>(ih, x_acc));
                args.push_back(sc.memory<void *>(ih, weights_acc));
                args.push_back(sc.memory<void *>(ih, y_acc));
                args.push_back(
                        with_bias ? sc.memory<void *>(ih, *bias_acc) : nullptr);
                args.push_back(with_scratchpad ? sc.memory<void *>(ih, *scratch_acc)
                                            : nullptr);
                args.push_back(pd()->impl_->using_transformed_filter()
                                ? sc.memory<void *>(ih, *filter_scratch_acc)
                                : nullptr);
                pd()->impl_->execute(handle, args);
            });
        });
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct miopen_convolution_bwd_weights_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public miopen_convolution_bwd_weights_pd_t {
        using miopen_convolution_bwd_weights_pd_t::
                miopen_convolution_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_convolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            bool ok = desc()->prop_kind == prop_kind::backward_weights;
            ok = ok && this->set_default_formats();
            ok = ok
                    && (utils::everyone_is(f32, src_md_.data_type,
                                diff_weights_md_.data_type,
                                diff_dst_md_.data_type)
                            || utils::everyone_is(f16, src_md_.data_type,
                                    diff_weights_md_.data_type,
                                    diff_dst_md_.data_type));

            ok = ok
                    && IMPLICATION(
                            desc()->alg_kind == dnnl_convolution_winograd,
                            ndims() < 5);
            ok = ok && memory_format_ok(&src_md_);
            ok = ok && memory_format_ok(&diff_weights_md_);
            ok = ok && memory_format_ok(&diff_dst_md_);
            if (with_bias()) {
                ok = ok && memory_format_ok(&diff_bias_md_);
                ok = ok && diff_bias_md_.data_type == diff_dst_md_.data_type;
            }
            if (!ok) return status::unimplemented;

            impl_.reset(new miopen_convolution_impl_bwd_weights_t());
            if (check_for_zero_dims()) { return impl_->init_zero_dims(this); };

            return impl_->init(engine, this);
        }

        std::shared_ptr<miopen_convolution_impl_base_t> impl_;

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
        bool with_scratchpad() const { return impl_->with_scratchpad(); }
    };

    ~miopen_convolution_bwd_weights_t() {}
    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->check_for_zero_dims()) { return execute_zero_dims(ctx); }
        return execute_convolution(
                ctx, pd()->with_bias(), pd()->with_scratchpad());
    }
    
    status_t execute_convolution(const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const{
        amd::sycl_hip_stream_t *hip_stream
            = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([&](::sycl::handler &cgh) {
            using scratch_acc_t = ::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::read_write>;
            auto x_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto weights_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_WEIGHTS);
            auto y_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
            std::shared_ptr<
                    ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::write>>
                    bias_acc;
            std::shared_ptr<scratch_acc_t> scratch_acc;
            std::shared_ptr<scratch_acc_t> filter_scratch_acc;
            if (with_scratchpad) {
                scratch_acc = std::make_shared<scratch_acc_t>(
                        utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                                ctx.get_scratchpad_grantor()
                                        .get_memory_storage(memory_tracking::names::
                                                        key_conv_cudnn_algo)
                                        .get())
                                ->buffer()
                                .get_access<::sycl::access::mode::read_write>(cgh));
            }
            if (with_bias) {
                bias_acc = std::make_shared<
                        ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::write>>(
                        CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_BIAS));
            }
            if (pd()->impl_->using_transformed_filter()) {
                filter_scratch_acc
                        = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
                                memory_tracking::names::key_conv_cudnn_filter));
            }

            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                        hip_stream->engine());
                auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
                auto handle = hip_stream->get_miopen_handle();

                std::vector<void *> args;
                args.push_back(sc.memory<void *>(ih, x_acc));
                args.push_back(sc.memory<void *>(ih, weights_acc));
                args.push_back(sc.memory<void *>(ih, y_acc));
                args.push_back(
                        with_bias ? sc.memory<void *>(ih, *bias_acc) : nullptr);
                args.push_back(with_scratchpad ? sc.memory<void *>(ih, *scratch_acc)
                                            : nullptr);
                args.push_back(pd()->impl_->using_transformed_filter()
                                ? sc.memory<void *>(ih, *filter_scratch_acc)
                                : nullptr);
                pd()->impl_->execute(handle, args);
            });
        });
    }
    status_t execute_zero_dims(const exec_ctx_t &ctx) const{
        amd::sycl_hip_stream_t *hip_stream
            = utils::downcast<amd::sycl_hip_stream_t *>(ctx.stream());

        return hip_stream->interop_task([&](::sycl::handler &cgh) {
            auto weights_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_WEIGHTS);
            std::shared_ptr<
                    ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::write>>
                    bias_acc;
            if (pd()->with_bias()) {
                bias_acc = std::make_shared<
                        ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::write>>(
                        CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_BIAS));
            }
            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(
                        hip_stream->engine());
                auto sc = hip_sycl_scoped_context_handler_t(sycl_engine);
                auto handle = hip_stream->get_miopen_handle();

                auto weights = sc.memory<void *>(ih, weights_acc);
                void *bias = nullptr;
                if (pd()->with_bias()) bias = sc.memory<void *>(ih, *bias_acc);
                pd()->impl_->execute_set_weights_bias(handle, weights, bias, 0.f);
            });
        });
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
