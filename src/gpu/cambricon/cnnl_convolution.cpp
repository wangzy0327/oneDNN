#include "gpu/cambricon/cnnl_convolution.hpp"
#include "gpu/cambricon/sycl_bang_scoped_context.hpp"
#include "gpu/cambricon/sycl_bang_stream.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"
#include <iostream>

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

status_t cnnl_convolution_fwd_t::execute_convolution(
        const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const {
    cambricon::sycl_bang_stream_t *bang_stream
            = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

    return bang_stream->interop_task([&](::sycl::handler &cgh) {
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

        // TODO: key_conv_cudnn_algo...
        if (with_scratchpad) {
            scratch_acc = std::make_shared<scratch_acc_t>(
                    utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                            ctx.get_scratchpad_grantor()
                                    .get_memory_storage(memory_tracking::names::
                                                    key_conv_cnnl)
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
            auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(
                    bang_stream->engine());
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = bang_stream->get_cnnl_handle();

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

status_t cnnl_convolution_bwd_data_t::execute_convolution(
        const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const {
    cambricon::sycl_bang_stream_t *bang_stream
            = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

    return bang_stream->interop_task([&](::sycl::handler &cgh) {
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
                                                    key_conv_cnnl)
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
            auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(
                    bang_stream->engine());
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = bang_stream->get_cnnl_handle();

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

status_t cnnl_convolution_bwd_weights_t::execute_zero_dims(
        const exec_ctx_t &ctx) const {
    cambricon::sycl_bang_stream_t *bang_stream
            = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

    return bang_stream->interop_task([&](::sycl::handler &cgh) {
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
            auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(
                    bang_stream->engine());
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = bang_stream->get_cnnl_handle();

            auto weights = sc.memory<void *>(ih, weights_acc);
            void *bias = nullptr;
            if (pd()->with_bias()) bias = sc.memory<void *>(ih, *bias_acc);
            pd()->impl_->execute_set_weights_bias(handle, weights, bias, 0.f);
        });
    });
}

status_t cnnl_convolution_bwd_weights_t::execute_convolution(
        const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const {
    cambricon::sycl_bang_stream_t *bang_stream
            = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

    return bang_stream->interop_task([&](::sycl::handler &cgh) {
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
                                                    key_conv_cnnl)
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
            auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(
                    bang_stream->engine());
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = bang_stream->get_cnnl_handle();

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

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl
