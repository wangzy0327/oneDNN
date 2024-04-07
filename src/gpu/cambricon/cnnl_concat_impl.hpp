#ifndef GPU_CAMBRICON_CNNL_CONCAT_IMPL_HPP
#define GPU_CAMBRICON_CNNL_CONCAT_IMPL_HPP

#include "cnnl.h"
#include "common/type_helpers.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_stream.hpp"
#include "gpu/cambricon/sycl_bang_scoped_context.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_concat_generic_t {
public:
    virtual status_t init(engine_t *engine, concat_pd_t *pd) = 0;

    virtual void execute(cnnlHandle_t handle, void** srcs, void* dst, void* workspace) const = 0;

    virtual ~cnnl_concat_generic_t() {
        for(int i=0; i<src_descs_.size(); i++)
        {
            CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, src_descs_[i]);
        }
        CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, dst_desc_);
    }
    
    bool with_scratchpad() const { return with_scratchpad_; }

    status_t init_scratchpad(engine_t *engine, concat_pd_t *pd)
    {
        auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto bang_stream = utils::downcast<sycl_bang_stream_t *>(service_stream);
        auto handle = bang_stream->get_cnnl_handle();
        
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetConcatWorkspaceSize, handle, concat_num_, &workspace_size_));

        if (workspace_size_ > 0)
        {
            with_scratchpad_ = true;
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_concat_cnnl,
                    workspace_size_, size_t(1));
        }
        return status::success;
    }

protected:
    int concat_num_;
    int axis_;
    std::vector<cnnlTensorDescriptor_t> src_descs_;
    cnnlTensorDescriptor_t dst_desc_;
    bool with_scratchpad_ = false;
    size_t workspace_size_ = 0;
    
    cnnlTensorLayout_t layout_ = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t data_type_;
    int ndims_;
    std::vector<std::vector<int>> src_dims_;
    int dst_dim_[DNNL_MAX_NDIMS];
};

// This structure is used when the memory format includes blocking
// struct cnnl_concat_ex_t : public cnnl_concat_generic_t {};

// This structure is used when the memory format does not include blocking
struct cnnl_concat_stride_t : public cnnl_concat_generic_t {
public:
    status_t init(engine_t *engine, concat_pd_t *pd) override {
        concat_num_ = pd->n_inputs();
        axis_ = pd->concat_dim();
        src_dims_.resize(concat_num_);
        src_descs_.resize(concat_num_);

        CHECK(convert_data_type(pd->src_md(0), &data_type_));
        ndims_ = pd->src_md(0)->ndims;
        // get_format(pd->src_md(0), layout_);

        for(int i=0; i<concat_num_; i++)
        {
            src_dims_[i].resize(ndims_);
            convert_dims(pd->src_md(i)->dims, src_dims_[i].data(), pd->src_md(i)->ndims);
            CHECK(create_and_set_tensor_descriptor(&src_descs_[i], layout_,
                    data_type_, ndims_, src_dims_[i].data()));
        }
        convert_dims(pd->dst_md()->dims, dst_dim_, pd->dst_md()->ndims);
        CHECK(create_and_set_tensor_descriptor(&dst_desc_, layout_,
                data_type_, ndims_, dst_dim_));

        // adjust_dim_for_dnn(dims_, pd->dst_md()->ndims, pd->src_md());

        init_scratchpad(engine, pd);
        return status::success;
    }

    void execute(cnnlHandle_t handle, void** srcs, void* dst, void* workspace) const override {
        CNNL_EXECUTE_FUNC(cnnlConcat, handle, concat_num_, axis_, src_descs_.data(), srcs, workspace, 
                workspace_size_, dst_desc_, dst);
    }

private:
    using cnnl_concat_generic_t::cnnl_concat_generic_t;
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
