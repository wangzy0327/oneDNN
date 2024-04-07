#ifndef GPU_CAMBRICON_CNNL_SPLIT_IMPL_HPP
#define GPU_CAMBRICON_CNNL_SPLIT_IMPL_HPP

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

// This structure is used when the memory format does not include blocking
struct cnnl_split_impl_t {
public:
    bool with_scratchpad() const { return with_scratchpad_; }

    status_t init_scratchpad(engine_t *engine, split_pd_t *pd)
    {
        auto &sycl_engine = *utils::downcast<sycl_bang_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto bang_stream = utils::downcast<sycl_bang_stream_t *>(service_stream);
        auto handle = bang_stream->get_cnnl_handle();
        
        CHECK(CNNL_EXECUTE_FUNC_S(cnnlGetSplitWorkspaceSize, handle, split_num_, &workspace_size_));

        if (workspace_size_ > 0)
        {
            with_scratchpad_ = true;
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_split_cnnl,
                    workspace_size_, size_t(1));
        }
        return status::success;
    }

    status_t init(engine_t *engine, split_pd_t *pd) {
        split_num_ = pd->n_outputs();
        axis_ = pd->split_dim();
        dst_dims_.resize(split_num_);
        dst_descs_.resize(split_num_);

        CHECK(convert_data_type(pd->dst_md(0), &data_type_));
        ndims_ = pd->src_md(0)->ndims;

        CHECK(format_to_cnnl_layout(pd->src_md()->format_tag, src_layout_));
        CHECK(format_to_cnnl_layout(pd->dst_md(0)->format_tag, dst_layout_));

        // configure dst cnnl_descs and dims
        for(int i=0; i<split_num_; i++)
        {
            dst_dims_[i].resize(ndims_);
            convert_dims(pd->dst_md(i)->dims, dst_dims_[i].data(), pd->dst_md(i)->ndims);
            CHECK(create_and_set_tensor_descriptor(&dst_descs_[i], dst_layout_,
                    data_type_, ndims_, dst_dims_[i].data()));
        }
        // configure src cnnl_desc and dim
        convert_dims(pd->src_md()->dims, src_dim_, pd->src_md()->ndims);
        CHECK(create_and_set_tensor_descriptor(&src_desc_, src_layout_,
                data_type_, ndims_, src_dim_));

        init_scratchpad(engine, pd);
        return status::success;
    }

    void execute(cnnlHandle_t handle, void** dsts, void* src, void* workspace) const {
        CNNL_EXECUTE_FUNC(cnnlSplit, handle, split_num_, axis_, src_desc_, src, workspace, 
                workspace_size_, dst_descs_.data(), dsts);
    }

    int split_num_;
    int axis_;
    std::vector<cnnlTensorDescriptor_t> dst_descs_;
    cnnlTensorDescriptor_t src_desc_;
    bool with_scratchpad_ = false;
    size_t workspace_size_ = 0;
    cnnlTensorLayout_t src_layout_, dst_layout_; 
    cnnlDataType_t data_type_;
    int ndims_;
    std::vector<std::vector<int>> dst_dims_;
    int src_dim_[DNNL_MAX_NDIMS];
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
