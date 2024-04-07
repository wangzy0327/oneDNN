#include "gpu/cambricon/cnnl_split.hpp"
#include "gpu/cambricon/sycl_bang_scoped_context.hpp"
#include "gpu/cambricon/sycl_bang_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

status_t cnnl_split_t::execute(const exec_ctx_t &ctx) const { 
    cambricon::sycl_bang_stream_t *bang_stream
            = utils::downcast<cambricon::sycl_bang_stream_t *>(ctx.stream());
    
    bool with_scratchpad = pd()->split_->with_scratchpad();

    return bang_stream->interop_task([&](::sycl::handler &cgh) {
        int n_outputs = pd()->n_outputs();
        auto dst_acc_0 = CTX_OUT_ACCESSOR(DNNL_ARG_MULTIPLE_DST);
        std::vector<decltype(dst_acc_0)> dst_accs;
        dst_accs.push_back(dst_acc_0);
        for(int i=1; i<n_outputs; i++)
        {
            auto dst_acc = CTX_OUT_ACCESSOR(DNNL_ARG_MULTIPLE_DST + i);
            dst_accs.push_back(dst_acc);
        }

        auto src_acc = CTX_IN_ACCESSOR(DNNL_ARG_SRC);

        using scratch_acc_t = ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::read_write>;
        std::shared_ptr<scratch_acc_t> scratch_acc;
        if(with_scratchpad){
            scratch_acc = std::make_shared<scratch_acc_t>(
            utils::downcast<sycl::sycl_buffer_memory_storage_t *>(
                    ctx.get_scratchpad_grantor()
                            .get_memory_storage(memory_tracking::names::
                                            key_split_cnnl)
                            .get())
                    ->buffer()
                    .get_access<::sycl::access::mode::read_write>(cgh));
        }

        compat::host_task(cgh, [=](const compat::interop_handle &ih) {
            auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(
                    bang_stream->engine());
            auto sc = bang_sycl_scoped_context_handler_t(sycl_engine);
            auto handle = bang_stream->get_cnnl_handle();

            std::vector<void*> dsts(n_outputs);
            for(int i=0; i<n_outputs; i++)
            {
                dsts[i] = sc.memory<uint8_t *>(ih, dst_accs[i]);
            }
            auto src = sc.memory<uint8_t *>(ih, src_acc);
            void* scratchpad = nullptr;
            if(with_scratchpad){
                scratchpad = sc.memory<void *>(ih, *scratch_acc);
            }
            pd()->split_->execute(handle, dsts.data(), src, scratchpad);
        });
    });
}

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl
