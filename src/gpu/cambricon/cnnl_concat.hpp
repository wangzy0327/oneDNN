#ifndef GPU_CAMBRICON_CNNL_CONCAT_HPP
#define GPU_CAMBRICON_CNNL_CONCAT_HPP

#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/concat_pd.hpp"
#include "gpu/cambricon/cnnl_concat_impl.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_concat_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public concat_pd_t {
        using concat_pd_t::concat_pd_t;
        DECLARE_COMMON_PD_T("bang:cnnl:any", cnnl_concat_t);

        // Function to verify data and memory format
        bool valid_data_n_mem_format() const {
            bool ok = true;
            auto datatype_0 = src_md(0)->data_type;
            if(datatype_0 == data_type::undef) return false;
            for(int i=0; i<n_inputs(); i++)
            {
                ok = ok && (src_md(i)->data_type == datatype_0);
                if(!ok) return false;
            }
            ok = ok && (dst_md()->data_type == datatype_0);
            if(!ok) return false;

            // TODO: recheck cnnl's support for blocking datatype
            for(int i=0; i<n_inputs(); i++)
            {
                if(src_md()->format_desc.blocking.inner_nblks>0)
                    return false;
            }
            if(dst_md()->format_desc.blocking.inner_nblks>0) return false;

            return ok;
        }

        bool check_scales_mask() const {
            // cnnl does not support scaling per dimension.
            if (attr()->output_scales_.mask_ != 0) { return false; }
            return true;
        }

        status_t init(engine_t *engine) {
            bool ok = (engine->kind() == engine_kind::gpu);
            ok = ok && valid_data_n_mem_format();
            ok = ok && check_scales_mask();

            if (!ok) return status::unimplemented;
            // if (has_different_block_size(src_md(), dst_md())) return status::unimplemented;
            concat_.reset(new cnnl_concat_stride_t());
            return concat_->init(engine, this);
        }
        
        std::shared_ptr<cnnl_concat_generic_t> concat_;
    private:
        static status_t create(concat_pd_t **concat_pd, engine_t *engine, 
                const primitive_attr_t *attr, const memory_desc_t *dst_md, 
                int n, int concat_dim, const memory_desc_t *src_mds) {
            // concat_pd_t(const primitive_attr_t *attr, const memory_desc_t *dst_md,
            // int n, int concat_dim, const memory_desc_t *src_mds)
            auto _pd = new pd_t(attr, dst_md, n, concat_dim, src_mds);
            if (_pd == nullptr) return status::out_of_memory;
            if (_pd->init(engine) != status::success) {
                delete _pd;
                return status::unimplemented;
            }
            // _pd->init_scratchpad_md();
            return safe_ptr_assign<concat_pd_t>(*concat_pd, _pd);
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
