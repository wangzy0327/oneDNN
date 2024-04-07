#ifndef GPU_CAMBRICON_CNNL_BATCH_NORMALIZATION_EXECUTOR_HPP
#define GPU_CAMBRICON_CNNL_BATCH_NORMALIZATION_EXECUTOR_HPP

#include "common/batch_normalization_pd.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "gpu/cambricon/cnnl_batch_normalization_impl.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_scoped_context.hpp"
#include "gpu/cambricon/sycl_bang_stream.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct bnorm_exec_base_t {
    virtual status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            const std::shared_ptr<cnnl_batch_normalization_impl_base_t>
                    bnorm_impl) const = 0;
    virtual ~bnorm_exec_base_t() = default;

protected:
    template <typename T, ::sycl::access::mode md, typename sc_t>
    void *mean_var_ptr(::sycl::accessor<T, 1, md> acc, sc_t &sc,
            const compat::interop_handle &ih) const {
        return sc.template memory<void *>(ih, acc);
    }

    template <typename sc_t>
    std::nullptr_t mean_var_ptr(std::nullptr_t acc, sc_t &,
            const compat::interop_handle &ih) const {
        return acc;
    }

    template <typename read_acc_t, typename write_acc_t, typename wkspace_st_t,
            typename float_acc_t, typename maybe_nullptr_t>
    void interop_task_fwd(
            std::shared_ptr<cnnl_batch_normalization_impl_base_t> bnorm_impl,
            engine_t *engine, ::sycl::handler &cgh,
            cambricon::sycl_bang_stream_t *bang_stream, read_acc_t src_acc,
            write_acc_t dst_acc, read_acc_t weight_acc, maybe_nullptr_t mean_acc,
            maybe_nullptr_t var_acc, float_acc_t scale_acc,
            float_acc_t bias_acc, wkspace_st_t wkspace_st, bool init_ss,
            bool init_mean_var) const {

        std::shared_ptr<
                ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::write>>
                wkspace_acc;
        if (!wkspace_st->is_null()) {
            wkspace_acc.reset(new ::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::write>(
                    utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                            wkspace_st)
                            ->buffer()
                            .template get_access<::sycl::access::mode::write>(
                                    cgh)));
        }

        maybe_init_mean_var(bang_stream, mean_acc, var_acc, init_mean_var);
        maybe_init_ss(bang_stream, scale_acc, bias_acc, init_ss);
        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = bang_stream->get_cnnl_handle();

            auto x = sc.memory<void *>(ih, src_acc);
            auto y = sc.memory<void *>(ih, dst_acc);
            auto weight = sc.memory<void *>(ih, weight_acc);
            auto mean = mean_var_ptr(mean_acc, sc, ih);
            auto var = mean_var_ptr(var_acc, sc, ih);
            auto scale = sc.memory<float *>(ih, scale_acc);
            auto bias = sc.memory<float *>(ih, bias_acc) + bnorm_impl->C();
            uint8_t *y_prime = nullptr, *save_mean = nullptr,
                    *save_var = nullptr;
            if (!wkspace_st->is_null()) {
                save_mean = sc.memory<uint8_t *>(ih, *wkspace_acc);
                save_var = save_mean + bnorm_impl->mean_var_size_bytes();
                y_prime = save_var + bnorm_impl->mean_var_size_bytes();
            }

            std::shared_ptr<bnorm_args_t> args(new bnorm_fwd_args_t(x, y, weight, mean,
                    var, scale, bias, y_prime, save_mean, save_var));

            bnorm_impl->execute(handle, args);
        });
    }

    template <typename read_acc_t, typename write_acc_t, typename ss_acc_t,
            typename d_ss_acc_t>
    void interop_task_bwd(
            std::shared_ptr<cnnl_batch_normalization_impl_base_t> bnorm_impl,
            engine_t *engine, ::sycl::handler &cgh,
            cambricon::sycl_bang_stream_t *bang_stream, read_acc_t src_acc,
            read_acc_t diff_dst_acc, write_acc_t diff_src_acc, read_acc_t weight_acc,
            ss_acc_t scale_acc, ss_acc_t bias_acc,
            d_ss_acc_t diff_scaleshift_acc, read_acc_t wkspace_acc,
            std::shared_ptr<::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::read_write,
                    sycl::compat::target_device>>
                    temp_relu_output,
            bool init_ss, bool init_mean_var) const {

        maybe_init_ss(bang_stream, scale_acc, bias_acc, init_ss);
        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = bang_stream->get_cnnl_handle();

            auto x = sc.memory<void *>(ih, src_acc);
            auto dy = sc.memory<void *>(ih, diff_dst_acc);
            auto dx = sc.memory<void *>(ih, diff_src_acc);
            auto weight = sc.memory<void *>(ih, weight_acc);
            auto scale = sc.memory<uint8_t *>(ih, scale_acc);
            auto bias = sc.memory<uint8_t *>(ih, bias_acc)
                    + (bnorm_impl->C() * sizeof(float));
            auto diff_scale = sc.memory<uint8_t *>(ih, diff_scaleshift_acc);
            auto diff_bias = diff_scale + (bnorm_impl->C() * sizeof(float));
            auto save_mean = sc.memory<uint8_t *>(ih, wkspace_acc);
            auto save_var = save_mean + bnorm_impl->mean_var_size_bytes();
            auto wkspace = save_var + bnorm_impl->mean_var_size_bytes();
            auto relu_dy = bnorm_impl->fuse_norm_relu()
                    ? sc.memory<void *>(ih, *temp_relu_output)
                    : nullptr;

            std::shared_ptr<bnorm_args_t> args(
                    new bnorm_bwd_args_t(x, dx, dy, weight, save_mean, save_var, scale,
                            bias, diff_scale, diff_bias, wkspace, relu_dy));

            bnorm_impl->execute(handle, args);
        });
    }

    template <typename T>
    void maybe_init_ss(
            cambricon::sycl_bang_stream_t *bang_stream, T, T, bool) const {}

    template <typename T>
    void maybe_init_ss(cambricon::sycl_bang_stream_t *bang_stream,
            ::sycl::accessor<T, 1, ::sycl::access::mode::write> scale_acc,
            ::sycl::accessor<T, 1, ::sycl::access::mode::write> bias_acc,
            bool init_ss) const {
        if (init_ss) {
            constexpr T scale_val = 1, bias_val = 0;
            bang_stream->interop_task([&](::sycl::handler &cgh) {
                cgh.fill(scale_acc, scale_val);
            });

            bang_stream->interop_task([&](::sycl::handler &cgh) {
                cgh.fill(bias_acc, bias_val);
            });
        }
    }

    // Handle the cases when mean and var are read-only accessors or nullptr
    template <typename T>
    void maybe_init_mean_var(
            cambricon::sycl_bang_stream_t *bang_stream, T, T, bool) const {}

    template <typename T>
    void maybe_init_mean_var(cambricon::sycl_bang_stream_t *bang_stream,
            ::sycl::accessor<T, 1, ::sycl::access::mode::write> mean_acc,
            ::sycl::accessor<T, 1, ::sycl::access::mode::write> var_acc,
            bool init_mean_var) const {
        if (init_mean_var) {
            constexpr T mean_var_val = 0;
            bang_stream->interop_task([&](::sycl::handler &cgh) {
                cgh.fill(mean_acc, mean_var_val);
            });

            bang_stream->interop_task([&](::sycl::handler &cgh) {
                cgh.fill(var_acc, mean_var_val);
            });
        }
    }
};

struct bnorm_exec_fwd_inf_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cnnl_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        cambricon::sycl_bang_stream_t *bang_stream
                = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

        auto wkspace_storage = bnorm_impl->is_training()
                ? ctx.output(DNNL_ARG_WORKSPACE)->memory_storage()
                : &memory_storage_t::empty_storage();

        auto n_channels = bnorm_impl->C();
        ::sycl::buffer<float> scaleshift_buff(n_channels * 2);
        return bang_stream->interop_task([&](::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto weight_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto scale_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::write>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::write>(
                            cgh, n_channels, n_channels);
            bool init_ss = true, init_mean_var = false;

            interop_task_fwd(bnorm_impl, engine, cgh, bang_stream, src_acc,
                    dst_acc, weight_acc,nullptr, nullptr, scale_acc, bias_acc,
                    wkspace_storage, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_inf_ss_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cnnl_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        cambricon::sycl_bang_stream_t *bang_stream
                = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

        auto wkspace_storage = bnorm_impl->is_training()
                ? ctx.output(DNNL_ARG_WORKSPACE)->memory_storage()
                : &memory_storage_t::empty_storage();

        auto scaleshift_buff
                = utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                        &CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT))
                          ->buffer();
        auto n_channels = bnorm_impl->C();

        return bang_stream->interop_task([&](::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto weight_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);

            auto scale_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::read>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::read>(
                            cgh, n_channels, n_channels);
            bool init_ss = false, init_mean_var = false;

            interop_task_fwd(bnorm_impl, engine, cgh, bang_stream, src_acc,
                    dst_acc, weight_acc, nullptr, nullptr, scale_acc, bias_acc,
                    wkspace_storage, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_inf_stats_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cnnl_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        cambricon::sycl_bang_stream_t *bang_stream
                = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

        auto wkspace_storage = bnorm_impl->is_training()
                ? ctx.output(DNNL_ARG_WORKSPACE)->memory_storage()
                : &memory_storage_t::empty_storage();

        auto n_channels = bnorm_impl->C();
        ::sycl::buffer<float> scaleshift_buff(n_channels * 2);
        return bang_stream->interop_task([&](::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto weight_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto mean_acc = CTX_IN_ACCESSOR(DNNL_ARG_MEAN);
            auto var_acc = CTX_IN_ACCESSOR(DNNL_ARG_VARIANCE);
            auto scale_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::write>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::write>(
                            cgh, n_channels, n_channels);
            bool init_ss = true, init_mean_var = false;

            interop_task_fwd(bnorm_impl, engine, cgh, bang_stream, src_acc,
                    dst_acc, weight_acc, mean_acc, var_acc, scale_acc, bias_acc,
                    wkspace_storage, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_inf_ss_stats_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cnnl_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        cambricon::sycl_bang_stream_t *bang_stream
                = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

        auto wkspace_storage = bnorm_impl->is_training()
                ? ctx.output(DNNL_ARG_WORKSPACE)->memory_storage()
                : &memory_storage_t::empty_storage();

        auto scaleshift_buff
                = utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                        &CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT))
                          ->buffer();
        auto n_channels = bnorm_impl->C();

        return bang_stream->interop_task([&](::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto weight_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
          
            auto mean_acc = CTX_IN_ACCESSOR(DNNL_ARG_MEAN);
            auto var_acc = CTX_IN_ACCESSOR(DNNL_ARG_VARIANCE);
            auto scale_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::read>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::read>(
                            cgh, n_channels, n_channels);
            bool init_ss = false, init_mean_var = false;

            interop_task_fwd(bnorm_impl, engine, cgh, bang_stream, src_acc,
                    dst_acc, weight_acc, mean_acc, var_acc, scale_acc, bias_acc,
                    wkspace_storage, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cnnl_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        cambricon::sycl_bang_stream_t *bang_stream
                = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

        auto wkspace_storage = bnorm_impl->is_training()
                ? ctx.output(DNNL_ARG_WORKSPACE)->memory_storage()
                : &memory_storage_t::empty_storage();

        auto n_channels = bnorm_impl->C();

        ::sycl::buffer<float> scaleshift_buff(n_channels * 2);
        return bang_stream->interop_task([&](::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto weight_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto mean_acc = CTX_OUT_ACCESSOR(DNNL_ARG_MEAN);
            auto var_acc = CTX_OUT_ACCESSOR(DNNL_ARG_VARIANCE);
            auto scale_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::write>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::write>(
                            cgh, n_channels, n_channels);
            bool init_ss = true, init_mean_var = true;

            interop_task_fwd(bnorm_impl, engine, cgh, bang_stream, src_acc,
                    dst_acc, weight_acc, mean_acc, var_acc, scale_acc, bias_acc,
                    wkspace_storage, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_fwd_ss_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cnnl_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        cambricon::sycl_bang_stream_t *bang_stream
                = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

        auto wkspace_storage = bnorm_impl->is_training()
                ? ctx.output(DNNL_ARG_WORKSPACE)->memory_storage()
                : &memory_storage_t::empty_storage();

        auto scaleshift_buff
                = utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                        &CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT))
                          ->buffer();
        auto n_channels = bnorm_impl->C();

        return bang_stream->interop_task([&](::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto weight_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);

            auto mean_acc = CTX_OUT_ACCESSOR(DNNL_ARG_MEAN);
            auto var_acc = CTX_OUT_ACCESSOR(DNNL_ARG_VARIANCE);
            auto scale_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::write>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::write>(
                            cgh, n_channels, n_channels);
            bool init_ss = false, init_mean_var = true;

            interop_task_fwd(bnorm_impl, engine, cgh, bang_stream, src_acc,
                    dst_acc, weight_acc, mean_acc, var_acc, scale_acc, bias_acc,
                    wkspace_storage, init_ss, init_mean_var);
        });
    }
};

struct bnorm_exec_bwd_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cnnl_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        cambricon::sycl_bang_stream_t *bang_stream
                = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

        auto n_channels = bnorm_impl->C();
        ::sycl::buffer<float> scaleshift_buff(n_channels * 2);
        ::sycl::buffer<float> diff_scaleshift_buff(n_channels * 2);

        return bang_stream->interop_task([&](::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
            auto diff_src_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
            auto weight_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto wkspace_acc = CTX_IN_ACCESSOR(DNNL_ARG_WORKSPACE);
            auto diff_scaleshift_acc
                    = diff_scaleshift_buff
                              .get_access<::sycl::access::mode::read>(cgh);
            auto scale_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::write>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::write>(
                            cgh, n_channels, n_channels);
            bool init_ss = true, init_mean_var = false;

            std::shared_ptr<::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::read_write,
                    sycl::compat::target_device>>
                    temp_relu_output = nullptr;
            if (bnorm_impl->fuse_norm_relu()) {
                temp_relu_output = std::make_shared<::sycl::accessor<uint8_t, 1,
                        ::sycl::access::mode::read_write,
                        sycl::compat::target_device>>(
                        CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
            }

            interop_task_bwd(bnorm_impl, engine, cgh, bang_stream, src_acc,
                    diff_dst_acc, diff_src_acc, weight_acc, scale_acc, bias_acc,
                    diff_scaleshift_acc, wkspace_acc, temp_relu_output, init_ss,
                    init_mean_var);
        });
    }
};

struct bnorm_exec_bwd_dw_ss_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cnnl_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        cambricon::sycl_bang_stream_t *bang_stream
                = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

        auto scaleshift_buff
                = utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                        &CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT))
                          ->buffer();

        auto n_channels = bnorm_impl->C();

        return bang_stream->interop_task([&](::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
            auto weight_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
            auto diff_src_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
            auto diff_scaleshift_acc
                    = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SCALE_SHIFT);
            auto scale_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::read>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::read>(
                            cgh, n_channels, n_channels);
            auto wkspace_acc = CTX_IN_ACCESSOR(DNNL_ARG_WORKSPACE);
            bool init_ss = false, init_mean_var = false;

            std::shared_ptr<::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::read_write,
                    sycl::compat::target_device>>
                    temp_relu_output = nullptr;
            if (bnorm_impl->fuse_norm_relu()) {
                temp_relu_output = std::make_shared<::sycl::accessor<uint8_t, 1,
                        ::sycl::access::mode::read_write,
                        sycl::compat::target_device>>(
                        CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
            }

            interop_task_bwd(bnorm_impl, engine, cgh, bang_stream, src_acc,
                    diff_dst_acc, diff_src_acc, weight_acc, scale_acc, bias_acc,
                    diff_scaleshift_acc, wkspace_acc, temp_relu_output, init_ss,
                    init_mean_var);
        });
    }
};

struct bnorm_exec_bwd_d_ss_t : public bnorm_exec_base_t {
    status_t execute(const exec_ctx_t &ctx, engine_t *engine,
            std::shared_ptr<cnnl_batch_normalization_impl_base_t> bnorm_impl)
            const override {
        cambricon::sycl_bang_stream_t *bang_stream
                = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

        auto scaleshift_buff
                = utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                        &CTX_IN_STORAGE(DNNL_ARG_SCALE_SHIFT))
                          ->buffer();
        auto n_channels = bnorm_impl->C();

        ::sycl::buffer<float> diff_scaleshift_buff(n_channels * 2);
        return bang_stream->interop_task([&](::sycl::handler &cgh) {
            auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
            auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
            auto diff_src_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
            auto weight_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);

            auto scale_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::read>(
                            cgh, n_channels, 0);
            auto bias_acc
                    = scaleshift_buff.get_access<::sycl::access::mode::read>(
                            cgh, n_channels, n_channels);
            auto diff_scaleshift_acc
                    = diff_scaleshift_buff
                              .get_access<::sycl::access::mode::read>(cgh);
            auto wkspace_acc = CTX_IN_ACCESSOR(DNNL_ARG_WORKSPACE);
            bool init_ss = false, init_mean_var = false;

            std::shared_ptr<::sycl::accessor<uint8_t, 1,
                    ::sycl::access::mode::read_write,
                    sycl::compat::target_device>>
                    temp_relu_output = nullptr;
            if (bnorm_impl->fuse_norm_relu()) {
                temp_relu_output = std::make_shared<::sycl::accessor<uint8_t, 1,
                        ::sycl::access::mode::read_write,
                        sycl::compat::target_device>>(
                        CTX_SCRATCH_ACCESSOR(memory_tracking::names::key_none));
            }

            interop_task_bwd(bnorm_impl, engine, cgh, bang_stream, src_acc,
                    diff_dst_acc, diff_src_acc, weight_acc, scale_acc, bias_acc,
                    diff_scaleshift_acc, wkspace_acc, temp_relu_output, init_ss,
                    init_mean_var);
        });
    }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
