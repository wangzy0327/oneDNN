#ifndef GPU_NVIDIA_CNNL_REORDER_IMPL_HPP
#define GPU_NVIDIA_CNNL_REORDER_IMPL_HPP

#include "common/type_helpers.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

struct cnnl_reorder_generic_t {
public:
    virtual status_t init(const reorder_pd_t *pd) = 0;

    virtual void execute(cnnlHandle_t handle, void *src, void *dst) const = 0;

    virtual ~cnnl_reorder_generic_t() {
        CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, src_desc_);
        CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, dst_desc_);
        CNNL_EXECUTE_FUNC_V(cnnlDestroyTransposeDescriptor, trans_desc);
    }

    int dst_offset_in_bytes() { return dst_offset_in_bytes_; }
    int src_offset_in_bytes() { return src_offset_in_bytes_; }

protected:
	cnnlTensorLayout_t layout_src;
    cnnlTensorLayout_t layout_dst;
    cnnlTransposeDescriptor_t trans_desc;
    cnnlDataType_t src_data_type_;
    cnnlDataType_t dst_data_type_;
    int ndims_;
    int dims_src[DNNL_MAX_NDIMS];
    int dims_dst[DNNL_MAX_NDIMS];
    std::vector<int> permute;
    // int permute[DNNL_MAX_NDIMS];
    bool is_dim5;
    cnnlTensorDescriptor_t src_desc_;
    cnnlTensorDescriptor_t dst_desc_;
    float alpha_, beta_;
    int dst_offset_in_bytes_ = 0;
    int src_offset_in_bytes_ = 0;
};

// This structure is used when the memory format includes blocking
// struct cnnl_reorder_ex_t : public cnnl_reorder_generic_t {};

// This structure is used when the memory format does not include blocking
struct cnnl_reorder_stride_t : public cnnl_reorder_generic_t {
public:
    status_t init(const reorder_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with creating
        // cnnl descriptors
        memory_desc_wrapper wrap(pd->src_md());
        if (wrap.size() == 0) { return status::success; }

        // Validity checks
        assert(pd->dst_md()->ndims == pd->src_md()->ndims);
        dst_offset_in_bytes_ = pd->dst_md()->offset0
                * types::data_type_size(pd->dst_md()->data_type);
        src_offset_in_bytes_ = pd->src_md()->offset0
                * types::data_type_size(pd->src_md()->data_type);

		// TODO: support scale
        alpha_ = pd->alpha();
        beta_ = pd->beta();
		assert(alpha_==1 && beta_==0);

        convert_dims(pd->dst_md()->dims, dims_dst, pd->dst_md()->ndims);
        convert_dims(pd->src_md()->dims, dims_src, pd->dst_md()->ndims);
        
        ndims_ = pd->src_md()->ndims;

        CHECK(convert_data_type(pd->src_md(), &src_data_type_));
        CHECK(convert_data_type(pd->dst_md(), &dst_data_type_));
        assert(src_data_type_ == dst_data_type_);

        bool ok = pd->dst_md()->ndims == ndims_;
        ok = ok && layout_n_permute(pd);
        if(!ok) return status::invalid_arguments;

		// Create and set tensor descriptor
        CHECK(create_and_set_tensor_descriptor(&src_desc_, layout_src, src_data_type_, ndims_, dims_src));
		CHECK(create_and_set_tensor_descriptor(&dst_desc_, layout_dst, dst_data_type_, ndims_, dims_dst));

        CNNL_EXECUTE_FUNC(cnnlCreateTransposeDescriptor, &trans_desc);
        CNNL_EXECUTE_FUNC(cnnlSetTransposeDescriptor, trans_desc, ndims_, permute.data());

        return status::success;
    }

    bool layout_n_permute(const reorder_pd_t *pd) {
        // cnnltensorlayout is not important(maybe not used in cnnlTranspose kernel)
        // the reorder operator will be done under the instruction of permute array
        layout_src = cnnlTensorLayout_t::CNNL_LAYOUT_ARRAY;
        layout_dst = cnnlTensorLayout_t::CNNL_LAYOUT_ARRAY;

        auto src_tag = pd->src_md()->format_tag;
        auto dst_tag = pd->dst_md()->format_tag;

        if(src_tag == format_tag::nchw){
            if(dst_tag == format_tag::nchw) 
                permute = {0, 1, 2, 3};
            else if(dst_tag == format_tag::nhwc)
                permute = {0, 2, 3, 1};
            else
                return false;
        }
        else if(src_tag == format_tag::nhwc){
            if(dst_tag == format_tag::nhwc) 
                permute = {0, 1, 2, 3};
            else if(dst_tag == format_tag::nchw)
                permute = {0, 3, 1, 2};
            else
                return false;
        }
        else if(src_tag == format_tag::goihw){
            if(dst_tag == format_tag::goihw) 
                permute = {0, 1, 2, 3, 4};
            else if(dst_tag == format_tag::gohwi)
                permute = {0, 1, 3, 4, 2};
            else
                return false;
        }
        else if(src_tag == format_tag::gohwi){
            if(dst_tag == format_tag::gohwi) 
                permute = {0, 1, 2, 3, 4};
            else if(dst_tag == format_tag::goihw)
                permute = {0, 1, 4, 2, 3};
            else
                return false;
        }
        else
            return false;
        return true;
    }

    void execute(cnnlHandle_t handle, void *src, void *dst) const override {
        CNNL_EXECUTE_FUNC(cnnlTranspose, handle, trans_desc, src_desc_, src, dst_desc_, dst);
    }

private:
    using cnnl_reorder_generic_t::cnnl_reorder_generic_t;
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
