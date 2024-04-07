#ifndef GPU_CAMBRICON_SYCL_BANG_ENGINE_HPP
#define GPU_CAMBRICON_SYCL_BANG_ENGINE_HPP

#include <cnnl.h>
#include <cnmlrt.h>
#include <cn_api.h>

//#include <cnnl/cnnl.h>

#include "common/stream.hpp"
#include "common/thread_local_storage.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"
#include "sycl/sycl_device_info.hpp"
#include "sycl/sycl_engine_base.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {


class bang_gpu_engine_impl_list_t {
public:
    // TODO: support sum primitive
    // static const dnnl::impl::impl_list_item_t *get_sum_implementation_list();
    
    static const impl_list_item_t *get_reorder_implementation_list(
        const memory_desc_t *src_md, const memory_desc_t *dst_md);
    static const dnnl::impl::impl_list_item_t *get_concat_implementation_list();
};

// bang_engine
class sycl_bang_engine_t : public dnnl::impl::sycl::sycl_engine_base_t {
public:
    using base_t = dnnl::impl::sycl::sycl_engine_base_t;
    
    sycl_bang_engine_t(engine_kind_t kind, const cl::sycl::device &dev,
            const cl::sycl::context &ctx, size_t index);
    sycl_bang_engine_t(const cl::sycl::device &dev,
            const cl::sycl::context &ctx, size_t index);
    
    status_t create_stream(stream_t **stream, unsigned flags) override;
    status_t create_stream(stream_t **stream, cl::sycl::queue &queue);

    const dnnl::impl::impl_list_item_t *get_reorder_implementation_list(
            const memory_desc_t *src_md,
            const memory_desc_t *dst_md) const override {
        return bang_gpu_engine_impl_list_t::get_reorder_implementation_list(src_md, dst_md);
    }
    const dnnl::impl::impl_list_item_t *
    get_concat_implementation_list() const override {
        return bang_gpu_engine_impl_list_t::get_concat_implementation_list();
    }
    const dnnl::impl::impl_list_item_t *
    get_sum_implementation_list() const override {
        // TODO:
        return NULL;
        // return bang_gpu_engine_impl_list_t::get_sum_implementation_list();
    }

    //void activate_stream_rocblas(stream_t *stream);
    void activate_stream_cnnl(stream_t *stream);
    
    const impl_list_item_t *get_implementation_list(
            const op_desc_t *) const override;
    
    CNcontext get_underlying_context() const; 
    //
    cnnlHandle_t *get_cnnl_handle();
    //cnnl_handle *get_cnnl_handle();

    const bool has_primary_context() const { return primary_context_; }
    device_id_t device_id() const override;

#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
protected:
    ~sycl_bang_engine_t() override = default;
#endif

private:
    status_t underlying_context_type();     // primary or non-primary
    status_t set_cnnl_handle();

    utils::thread_local_storage_t<
            std::unique_ptr<cnnlHandle_t, void (*)(cnnlHandle_t *)>>
            cnnl_handle_;

    bool primary_context_;
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
