#ifndef GPU_CAMBRICON_CNNL_REORDER_HPP
#define GPU_CAMBRICON_CNNL_REORDER_HPP

#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/reorder_pd.hpp"
#include "gpu/cambricon/cnnl_reorder_impl.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_reorder_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public reorder_pd_t {
        using reorder_pd_t::reorder_pd_t;
        DECLARE_COMMON_PD_T("bang:cnnl:any", cnnl_reorder_t);

        // Function to verify data and memory format
        bool valid_data_n_mem_format() const {
            bool ok = utils::one_of(src_md()->data_type, data_type::s8,
                              data_type::f16, data_type::f32)
                    && utils::one_of(dst_md()->data_type, data_type::s8,
                            data_type::f16, data_type::f32);

            // Nvidia only supports blocking for Int8
            if (!utils::one_of(src_md()->data_type, data_type::s8)
                    && src_md()->format_desc.blocking.inner_nblks > 0)
                return false;
            if (!utils::one_of(dst_md()->data_type, data_type::s8)
                    && dst_md()->format_desc.blocking.inner_nblks > 0)
                return false;

            // Nvidia supports blocking only on channel dimension C
            if (dst_md()->format_desc.blocking.inner_nblks > 1
                    || src_md()->format_desc.blocking.inner_nblks > 1)
                return false;
            if (utils::one_of(src_md()->data_type, data_type::s8)
                    && src_md()->format_desc.blocking.inner_nblks == 1) {
                ok = ok && memory_desc_matches_nchw_vect_c(src_md());
            }
            int blks = dst_md()->format_desc.blocking.inner_nblks;
            if (utils::one_of(dst_md()->data_type, data_type::s8)
                    && blks == 1) {
                ok = ok && memory_desc_matches_nchw_vect_c(dst_md());
            }
            return ok;
        }

        bool check_scales_mask() const {
            // cuDNN does not support scaling per dimension.
            if (attr()->output_scales_.mask_ != 0) { return false; }
            return true;
        }

        status_t init(
                engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
            bool ok = true && (engine == dst_engine)
                    && (src_engine->kind() == engine_kind::gpu)
                    && valid_data_n_mem_format() && check_scales_mask();
            if (!ok) return status::unimplemented;
            if (has_different_block_size(src_md(), dst_md())) {
                assert("not support in cnnl" && 0);
                // reorder_.reset(new cnnl_reorder_ex_t());
            } else {
                reorder_.reset(new cnnl_reorder_stride_t());
            }

            return reorder_->init(this);
        }
        std::shared_ptr<cnnl_reorder_generic_t> reorder_;

    private:
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            auto _pd = new pd_t(attr, src_engine->kind(), src_md,
                    dst_engine->kind(), dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            if (_pd->init(engine, src_engine, dst_engine) != status::success) {
                delete _pd;
                return status::unimplemented;
            }
            _pd->init_scratchpad_md();
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }
        friend dnnl::impl::impl_list_item_t;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
