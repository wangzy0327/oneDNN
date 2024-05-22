#include "gpu/cambricon/cnnl_pooling.hpp"
#include "gpu/cambricon/sycl_bang_scoped_context.hpp"
#include "gpu/cambricon/sycl_bang_stream.hpp"
#include "sycl/sycl_buffer_memory_storage.hpp"
#include "sycl/sycl_memory_storage_helper.hpp"

#include <CL/sycl.hpp>

#include "common/nstl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

status_t cnnl_pooling_fwd_t::execute(const exec_ctx_t &ctx) const {
    // If dst is empty, do nothing
    memory_desc_wrapper dst_wrap(pd()->dst_md());
    if (dst_wrap.size() == 0) return status::success;
    cambricon::sycl_bang_stream_t *bang_stream
            = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());
    memory_desc_wrapper src_wrap(pd()->src_md());

    bool with_scratchpad = pd()->pooling_impl_->with_scratchpad();
    
    // If src is empty and dst is not, fill dst with
    // numeric_limits<dt>::lowest() to match the other backends' behaviour
    if (src_wrap.size() == 0 && dst_wrap.size() != 0) {
        return bang_stream->interop_task([&](::sycl::handler &cgh) {
            // auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
            auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);          

            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(
                        bang_stream->engine());
                auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);

                // auto dst = sc.memory<void *>(ih, dst_acc);
                auto dst = arg_dst.get_native_pointer(ih);

                if (dst_wrap.data_type() == data_type_t::dnnl_f32) {
                    auto val = nstl::numeric_limits<float>::lowest();
                    cnMemsetD32Async(reinterpret_cast<CNaddr>(dst),
                            reinterpret_cast<int &>(val), dst_wrap.nelems(),
                            bang_stream->get_underlying_stream());
                } else if (dst_wrap.data_type() == data_type_t::dnnl_f16) {
                    float16_t val = nstl::numeric_limits<float16_t>::lowest();
                    cnMemsetD16Async(reinterpret_cast<CNaddr>(dst),
                            reinterpret_cast<unsigned short &>(val),
                            dst_wrap.nelems(),
                            bang_stream->get_underlying_stream());
                } else if (dst_wrap.data_type() == data_type_t::dnnl_s8) {
                    auto val = nstl::numeric_limits<int8_t>::lowest();
                    cnMemsetD8Async(reinterpret_cast<CNaddr>(dst),
                            reinterpret_cast<unsigned char &>(val),
                            dst_wrap.nelems(),
                            bang_stream->get_underlying_stream());
                }
            });
        });
    }

    return bang_stream->interop_task([&](::sycl::handler &cgh) {
        // auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);
        // auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DST);
        auto arg_src = CTX_IN_SYCL_MEMORY(DNNL_ARG_SRC);
        auto arg_dst = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DST);        
        
        // using scratch_acc_t = ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::read_write>;
        // std::shared_ptr<scratch_acc_t> scratch_acc;
        using scratch_arg_t = impl::sycl::sycl_memory_arg_t<::sycl::access::mode::read_write>;
        std::shared_ptr<scratch_arg_t> arg_scratch_ptr;        
        if(with_scratchpad){
            // scratch_acc = std::make_shared<scratch_acc_t>(
            // utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
            //         ctx.get_scratchpad_grantor()
            //                 .get_memory_storage(memory_tracking::names::
            //                                 key_pool_cnnl)
            //                 .get())
            //         ->buffer()
            //         .get_access<::sycl::access::mode::read_write>(cgh));
            arg_scratch_ptr = std::make_shared<scratch_arg_t>(impl::sycl::sycl_memory_arg_t<
                    ::sycl::access::mode::read_write>(ctx.get_scratchpad_grantor().get_memory_storage(memory_tracking::names::key_pool_cnnl).get(), cgh));
        }

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(
                    bang_stream->engine());
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = bang_stream->get_cnnl_handle();

            // auto x = sc.memory<void *>(ih, src_acc);
            // auto y = sc.memory<void *>(ih, dst_acc);
            auto x = arg_src.get_native_pointer(ih);
            auto y = arg_dst.get_native_pointer(ih);            
            void* scratchpad = nullptr;
            if(with_scratchpad){
                // scratchpad = sc.memory<void *>(ih, *scratch_acc);
                scratchpad = arg_scratch_ptr->get_native_pointer(ih);
            }
            pd()->pooling_impl_->execute(handle, x, y, scratchpad);
        });
    });
}

status_t cnnl_pooling_bwd_t::execute(const exec_ctx_t &ctx) const {
    if (has_zero_dims(pd()->diff_src_md()->dims, pd()->diff_src_md()->ndims)
            || has_zero_dims(
                    pd()->diff_dst_md()->dims, pd()->diff_dst_md()->ndims)) {
        return status::success;
    }

    memory_desc_wrapper wrap(pd()->diff_src_md());
    if (wrap.size() == 0) { return status::success; }
    const auto dst_offset_bytes = wrap.size();

    cambricon::sycl_bang_stream_t *bang_stream
            = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());

    return bang_stream->interop_task([&](::sycl::handler &cgh) {
        // auto diff_src_acc = CTX_OUT_ACCESSOR(DNNL_ARG_DIFF_SRC);
        // auto diff_dst_acc = CTX_IN_ACCESSOR(DNNL_ARG_DIFF_DST);
        // // not used
        // auto wkspace_acc = CTX_IN_ACCESSOR(DNNL_ARG_WORKSPACE);
        auto arg_diff_src = CTX_OUT_SYCL_MEMORY(DNNL_ARG_DIFF_SRC);
        auto arg_diff_dst = CTX_IN_SYCL_MEMORY(DNNL_ARG_DIFF_DST);
        // not used
        auto arg_wkspace = CTX_IN_SYCL_MEMORY(DNNL_ARG_WORKSPACE);        

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(
                    bang_stream->engine());
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = bang_stream->get_cnnl_handle();

            // auto dx = sc.memory<void *>(ih, diff_src_acc);
            // auto dy = sc.memory<void *>(ih, diff_dst_acc);
            // auto ws = sc.memory<uint8_t *>(ih, wkspace_acc);
            auto dx = arg_diff_src.get_native_pointer(ih);
            auto dy = arg_diff_dst.get_native_pointer(ih);
            auto ws = arg_wkspace.get_native_pointer(ih);            
            // auto ws_y = ws + dst_offset_bytes;
            pd()->pooling_impl_->execute(handle, dx, dy, ws);
        });
    });
}

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl
