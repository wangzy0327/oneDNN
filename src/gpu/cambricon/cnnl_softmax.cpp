#include "gpu/cambricon/cnnl_softmax.hpp"
#include "gpu/cambricon/sycl_bang_scoped_context.hpp"
#include "gpu/cambricon/sycl_bang_stream.hpp"
#include "sycl/sycl_buffer_memory_storage.hpp"
#include "sycl/sycl_memory_storage_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

status_t cnnl_softmax_fwd_t::execute(const exec_ctx_t &ctx) const {
    if (memory_desc_wrapper(pd()->desc()->data_desc).has_zero_dim())
        return status::success;

    cambricon::sycl_bang_stream_t *bang_stream
            = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

    return bang_stream->interop_task([&](::sycl::handler &cgh) {
        // auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
        // auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
        auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);        

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            std::vector<void *> args;
            auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(
                    bang_stream->engine());
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = bang_stream->get_cnnl_handle();

            // args.push_back(sc.memory<void *>(ih, src_acc));
            // args.push_back(sc.memory<void *>(ih, dst_acc));
            args.push_back(arg_src.get_native_pointer(ih));
            args.push_back(arg_dst.get_native_pointer(ih));            

            pd()->softmax_impl_->execute(handle, args.data(), args.size());
        });
    });
}

status_t cnnl_softmax_bwd_t::execute(const exec_ctx_t &ctx) const {
    if (memory_desc_wrapper(pd()->desc()->diff_desc).has_zero_dim())
        return status::success;

    cambricon::sycl_bang_stream_t *bang_stream
            = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

    return bang_stream->interop_task([&](::sycl::handler &cgh) {
        // auto dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DST);
        // auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
        // auto diff_src_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
        auto arg_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DST);
        auto arg_diff_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);
        auto arg_diff_src = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_SRC);        

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            std::vector<void *> args;
            auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(
                    bang_stream->engine());
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = bang_stream->get_cnnl_handle();

            // args.push_back(sc.memory<void *>(ih, dst_acc));
            // args.push_back(sc.memory<void *>(ih, diff_dst_acc));
            // args.push_back(sc.memory<void *>(ih, diff_src_acc));
            args.push_back(arg_dst.get_native_pointer(ih));
            args.push_back(arg_diff_dst.get_native_pointer(ih));
            args.push_back(arg_diff_src.get_native_pointer(ih));


            pd()->softmax_impl_->execute(handle, args.data(), args.size());
        });
    });
}

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl
