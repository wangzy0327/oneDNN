#include "gpu/cambricon/cnnl_convolution.hpp"
#include "gpu/cambricon/sycl_bang_scoped_context.hpp"
#include "gpu/cambricon/sycl_bang_stream.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"
#include "sycl/sycl_memory_storage_helper.hpp"
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
        using scratch_arg_t = impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read_write>;
        // auto x_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
        // auto weights_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
        // auto y_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
        auto arg_x = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_weights = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
        auto arg_y = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);        
        std::shared_ptr<
                impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read>>
                arg_bias_ptr;
        std::shared_ptr<scratch_arg_t> arg_scratch_ptr;
        std::shared_ptr<scratch_arg_t> arg_filter_scratch_ptr;
        std::shared_ptr<scratch_arg_t> arg_temp_dst_ptr;
        std::shared_ptr<scratch_arg_t> arg_temp_reorder_ptr;

        const bool use_temp_dst = pd()->use_temp_dst();

        // TODO: key_conv_cudnn_algo...
        if (with_scratchpad) {
        //     scratch_acc = std::make_shared<scratch_acc_t>(
        //             utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
        //                     ctx.get_scratchpad_grantor()
        //                             .get_memory_storage(memory_tracking::names::
        //                                             key_conv_cnnl)
        //                             .get())
        //                     ->buffer()
        //                     .get_access<::sycl::access::mode::read_write>(cgh));
            arg_scratch_ptr = std::make_shared<scratch_arg_t>(impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read_write>(ctx.get_scratchpad_grantor().get_memory_storage(memory_tracking::names::key_pool_cnnl).get(), cgh));                            
        }
        if (with_bias) {
        //     bias_acc = std::make_shared<
        //             ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::read>>(
        //             CTX_IN_ACCESSOR(DNNL_ARG_BIAS));
            arg_bias_ptr = std::make_shared<impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read>>(CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS));
        }

        if (pd()->impl_->using_transformed_filter()) {
        //     filter_scratch_acc
        //             = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
        //                     memory_tracking::names::key_conv_cudnn_filter));
            arg_filter_scratch_ptr = std::make_shared<scratch_arg_t>(CTX_SCRATCH_SYCL_MEMORY(
                            memory_tracking::names::key_conv_cudnn_filter));                           
        }
        if (use_temp_dst) {
        //     temp_dst_acc = std::make_shared<scratch_acc_t>(
        //             buffer(scratch_storage.get())
        //                     .get_access<::sycl::access::mode::read_write>(cgh));
        //     temp_reorder_acc = std::make_shared<scratch_acc_t>(
        //             buffer(scratch_storage_2.get())
        //                     .get_access<::sycl::access::mode::read_write>(cgh));
            arg_temp_dst_ptr = std::make_shared<scratch_arg_t>(impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read_write>(buffer(scratch_storage.get()),cgh));
           arg_temp_reorder_ptr = std::make_shared<scratch_arg_t>(impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read_write>(buffer(scratch_storage_2.get()),cgh));                                               
        }
        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(
                    bang_stream->engine());
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = bang_stream->get_cnnl_handle();

            std::vector<void *> args;
        //     args.push_back(sc.memory<void *>(ih, x_acc));
        //     args.push_back(sc.memory<void *>(ih, weights_acc));
        //     args.push_back(sc.memory<void *>(ih, y_acc));
            args.push_back(arg_x.get_native_pointer(ih));
            args.push_back(arg_weights.get_native_pointer(ih));
            args.push_back(arg_y.get_native_pointer(ih));            
            args.push_back(
                    with_bias ? arg_bias_ptr->get_native_pointer(ih) : nullptr);
            args.push_back(with_scratchpad ? arg_scratch_ptr->get_native_pointer(ih)
                                           : nullptr);
            args.push_back(pd()->impl_->using_transformed_filter()
                            ? arg_filter_scratch_ptr->get_native_pointer(ih)
                            : nullptr);
            args.push_back(use_temp_dst ? arg_temp_dst_ptr->get_native_pointer(ih)
                                        : nullptr);
            args.push_back(use_temp_dst
                            ? arg_temp_reorder_ptr->get_native_pointer(ih)
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
        // using scratch_acc_t = ::sycl::accessor<uint8_t, 1,
        //         ::sycl::access::mode::read_write>;
        using scratch_arg_t = impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read_write>;                
        // auto x_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
        // auto weights_acc = CTX_IN_ACCESSOR(DNNL_ARG_WEIGHTS);
        // auto y_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
        auto arg_x = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_SRC);
        auto arg_weights = CTX_IN_SYCL_MEMORY(DNNL_ARG_WEIGHTS);
        auto arg_y = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);
        std::shared_ptr<
                impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read>>
                arg_bias_ptr;      
        std::shared_ptr<scratch_arg_t> arg_scratch_ptr;
        std::shared_ptr<scratch_arg_t> arg_filter_scratch_ptr;
        if (with_scratchpad) {
        //     scratch_acc = std::make_shared<scratch_acc_t>(
        //             utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
        //                     ctx.get_scratchpad_grantor()
        //                             .get_memory_storage(memory_tracking::names::
        //                                             key_conv_cnnl)
        //                             .get())
        //                     ->buffer()
        //                     .get_access<::sycl::access::mode::read_write>(cgh));
            arg_scratch_ptr = std::make_shared<scratch_arg_t>(impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read_write>(ctx.get_scratchpad_grantor().get_memory_storage(memory_tracking::names::key_conv_cnnl).get(), cgh));        
        }
        if (with_bias) {
        //     bias_acc = std::make_shared<
        //             ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::read>>(
        //             CTX_IN_ACCESSOR(DNNL_ARG_BIAS));
            arg_bias_ptr = std::make_shared<impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read>>(CTX_IN_SYCL_MEMORY(DNNL_ARG_BIAS));        
        }
        if (pd()->impl_->using_transformed_filter()) {
        //     filter_scratch_acc
        //             = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
        //                     memory_tracking::names::key_conv_cudnn_filter));
            arg_filter_scratch_ptr = std::make_shared<scratch_arg_t>(CTX_SCRATCH_SYCL_MEMORY(
                            memory_tracking::names::key_conv_cudnn_filter));                            
        }
        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(
                    bang_stream->engine());
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = bang_stream->get_cnnl_handle();

            std::vector<void *> args;
        //     args.push_back(sc.memory<void *>(ih, x_acc));
        //     args.push_back(sc.memory<void *>(ih, weights_acc));
        //     args.push_back(sc.memory<void *>(ih, y_acc));
            args.push_back(arg_x.get_native_pointer(ih));
            args.push_back(arg_weights.get_native_pointer(ih));
            args.push_back(arg_y.get_native_pointer(ih));
            args.push_back(
                    with_bias ? arg_bias_ptr->get_native_pointer(ih) : nullptr);
            args.push_back(with_scratchpad ? arg_scratch_ptr->get_native_pointer(ih)
                                           : nullptr);
            args.push_back(pd()->impl_->using_transformed_filter()
                            ? arg_filter_scratch_ptr->get_native_pointer(ih)
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
        // auto weights_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_WEIGHTS);
        auto arg_weights = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_WEIGHTS);
        // std::shared_ptr<
        //         ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::write>>
        //         bias_acc;
        std::shared_ptr<impl::sycl::sycl_memory_arg_t<::sycl::access::mode::write>> arg_bias_ptr;                
        if (pd()->with_bias()) {
        //     bias_acc = std::make_shared<
        //             ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::write>>(
        //             CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_BIAS));
            arg_bias_ptr = std::make_shared<impl::sycl::sycl_memory_arg_t<::sycl::access::mode::write>>(CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_BIAS));        
        }
        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(
                    bang_stream->engine());
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = bang_stream->get_cnnl_handle();

        //     auto weights = sc.memory<void *>(ih, weights_acc);
            auto weights = arg_weights.get_native_pointer(ih);
            void *bias = nullptr;
            if (pd()->with_bias()) 
                bias = arg_bias_ptr->get_native_pointer(ih);
                // bias = sc.memory<void *>(ih, *bias_acc);
            pd()->impl_->execute_set_weights_bias(handle, weights, bias, 0.f);
        });
    });
}

status_t cnnl_convolution_bwd_weights_t::execute_convolution(
        const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const {
    cambricon::sycl_bang_stream_t *bang_stream
            = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

    return bang_stream->interop_task([&](::sycl::handler &cgh) {
        // using scratch_acc_t = ::sycl::accessor<uint8_t, 1,
        //         ::sycl::access::mode::read_write>;
        using scratch_arg_t = impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read_write>; 
        // auto x_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
        // auto weights_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_WEIGHTS);
        // auto y_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
        auto arg_x = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_weights = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_WEIGHTS);
        auto arg_y = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);        
        // std::shared_ptr<
        //         ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::write>>
        //         bias_acc;
        // std::shared_ptr<scratch_acc_t> scratch_acc;
        // std::shared_ptr<scratch_acc_t> filter_scratch_acc;
        std::shared_ptr<
                impl::sycl::sycl_memory_arg_t<::sycl::access::mode::write>>
                arg_bias_ptr;
        std::shared_ptr<scratch_arg_t> arg_scratch_ptr;
        std::shared_ptr<scratch_arg_t> arg_filter_scratch_ptr;        
        if (with_scratchpad) {
        //     scratch_acc = std::make_shared<scratch_acc_t>(
        //             utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
        //                     ctx.get_scratchpad_grantor()
        //                             .get_memory_storage(memory_tracking::names::
        //                                             key_conv_cnnl)
        //                             .get())
        //                     ->buffer()
        //                     .get_access<::sycl::access::mode::read_write>(cgh));
            arg_scratch_ptr = std::make_shared<scratch_arg_t>(impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read_write>(ctx.get_scratchpad_grantor().get_memory_storage(memory_tracking::names::key_conv_cnnl).get(), cgh));  
        }
        if (with_bias) {
        //     bias_acc = std::make_shared<
        //             ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::write>>(
        //             CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_BIAS));
            arg_bias_ptr = std::make_shared<impl::sycl::sycl_memory_arg_t<::sycl::access::mode::write>>(CTX_OUT_SYCL_MEMORY(DNNL_ARG_BIAS));         
        }
        if (pd()->impl_->using_transformed_filter()) {
        //     filter_scratch_acc
        //             = std::make_shared<scratch_acc_t>(CTX_SCRATCH_ACCESSOR(
        //                     memory_tracking::names::key_conv_cudnn_filter));
            arg_filter_scratch_ptr = std::make_shared<scratch_arg_t>(CTX_SCRATCH_SYCL_MEMORY(
                            memory_tracking::names::key_conv_cudnn_filter));         
        }

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(
                    bang_stream->engine());
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = bang_stream->get_cnnl_handle();

            std::vector<void *> args;
        //     args.push_back(sc.memory<void *>(ih, x_acc));
        //     args.push_back(sc.memory<void *>(ih, weights_acc));
        //     args.push_back(sc.memory<void *>(ih, y_acc));
            args.push_back(arg_x.get_native_pointer(ih));
            args.push_back(arg_weights.get_native_pointer(ih));
            args.push_back(arg_y.get_native_pointer(ih));            
            args.push_back(
                    with_bias ? arg_bias_ptr->get_native_pointer(ih) : nullptr);
            args.push_back(with_scratchpad ? arg_scratch_ptr->get_native_pointer(ih)
                                           : nullptr);
            args.push_back(pd()->impl_->using_transformed_filter()
                            ? arg_filter_scratch_ptr->get_native_pointer(ih)
                            : nullptr);
            pd()->impl_->execute(handle, args);
        });
    });
}

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl
