#include "gpu/amd/sycl_hip_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_ref_sum_t : public ::dnnl::impl::gpu::ocl::ref_sum_t {
    using base_t = dnnl::impl::gpu::ocl::ref_sum_t;
    using base_t::base_t;
    using base_pd_t = base_t::pd_t;

    struct pd_t : public base_pd_t {

        using base_pd_t::base_pd_t;

        DECLARE_SUM_PD_T("ref:any", miopen_ref_sum_t);
        // This function can be used for backend that does not support
        // blocking on f32, so it can convert the blocked format to nchw. Since
        // the final destination will preserve the blocking, the last reorder
        // to put the accumulated result to the final output will add the
        // blocking back.
        void define_dst_acc_md() override {
            dst_acc_md_ = dst_md_;
            dst_acc_md_.data_type = dnnl_f32;
            if ((dst_md_.data_type == data_type::s8)
                    && (memory_desc_matches_nchw_vect_c(&dst_md_))) {
                dst_acc_md_.format_desc.blocking.inner_nblks = 0;
                dst_acc_md_.format_desc.blocking.inner_idxs[0] = 0;
                dst_acc_md_.format_desc.blocking.inner_blks[0] = 0;
                dst_acc_md_.format_desc.blocking.strides[dst_acc_md_.ndims - 1]
                        = 1;
                for (int i = dst_acc_md_.ndims - 2; i >= 0; i--) {
                    dst_acc_md_.format_desc.blocking.strides[i]
                            = dst_acc_md_.format_desc.blocking.strides[i + 1]
                            * dst_acc_md_.dims[i + 1];
                }
            }
        }
    };
}

namespace {

const impl_list_item_t hip_sum_impl_list[] = {
        // impl_list_item_t::sum_type_deduction_helper_t<miopen_ref_sum_t::pd_t>(),
        nullptr
        };
} // namespace

const impl_list_item_t *
hip_gpu_engine_impl_list_t::get_sum_implementation_list() {
    return hip_sum_impl_list;
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl