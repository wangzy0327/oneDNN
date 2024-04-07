/*******************************************************************************
* engine_t是上下文（类似进程控制块PCB）和primitive实例列表的保管者
*******************************************************************************/
#ifndef GPU_AMD_SYCL_HIP_ENGINE_HPP
#define GPU_AMD_SYCL_HIP_ENGINE_HPP

#include <rocblas.h>
#include <miopen/miopen.h>

#include "common/stream.hpp"
#include "common/thread_local_storage.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"
#include "sycl/sycl_device_info.hpp"
#include "sycl/sycl_engine_base.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

class hip_gpu_engine_impl_list_t {
public:
    static const impl_list_item_t *get_reorder_implementation_list(
        const memory_desc_t *src_md, const memory_desc_t *dst_md){
            return nullptr;
        };
    static const dnnl::impl::impl_list_item_t *get_concat_implementation_list();
    static const dnnl::impl::impl_list_item_t *get_sum_implementation_list(){
        return nullptr;
    };
};

// hip_engine
class sycl_hip_engine_t : public dnnl::impl::sycl::sycl_engine_base_t {
public:
    using base_t = dnnl::impl::sycl::sycl_engine_base_t;
    
    sycl_hip_engine_t(engine_kind_t kind, const cl::sycl::device &dev,
            const cl::sycl::context &ctx, size_t index);
    sycl_hip_engine_t(const cl::sycl::device &dev,
            const cl::sycl::context &ctx, size_t index);
    
    status_t create_stream(stream_t **stream, unsigned flags) override;
    status_t create_stream(stream_t **stream, cl::sycl::queue &queue);

    const dnnl::impl::impl_list_item_t *get_reorder_implementation_list(
            const memory_desc_t *src_md,
            const memory_desc_t *dst_md) const override {
        return hip_gpu_engine_impl_list_t::get_reorder_implementation_list(
                src_md, dst_md);
    }
    const dnnl::impl::impl_list_item_t *
    get_concat_implementation_list() const override {
        return hip_gpu_engine_impl_list_t::get_concat_implementation_list();
    }
    const dnnl::impl::impl_list_item_t *
    get_sum_implementation_list() const override {
        return hip_gpu_engine_impl_list_t::get_sum_implementation_list();
    }


    void activate_stream_rocblas(stream_t *stream);
    void activate_stream_miopen(stream_t *stream);
    
    const impl_list_item_t *get_implementation_list(
            const op_desc_t *) const override;
    
    hipCtx_t get_underlying_context() const; 
    miopenHandle_t *get_miopen_handle();
    rocblas_handle *get_rocblas_handle();

    const bool has_primary_context() const { return primary_context_; }
    device_id_t device_id() const override;

#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
protected:
    ~sycl_hip_engine_t() override = default;
#endif

private:
    status_t underlying_context_type();     // primary or non-primary
    status_t set_miopen_handle();
    status_t set_rocblas_handle();

    utils::thread_local_storage_t<
            std::unique_ptr<miopenHandle_t, void (*)(miopenHandle_t *)>>
            miopen_handle_;
    utils::thread_local_storage_t<
            std::unique_ptr<rocblas_handle, void (*)(rocblas_handle *)>>
            rocblas_handle_;

    bool primary_context_;
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif