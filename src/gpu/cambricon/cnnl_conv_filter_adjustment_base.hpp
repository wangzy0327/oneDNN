#ifndef GPU_CAMBRICON_CNNL_CONV_FILTER_ADJUSTMENT_BASE_HPP
#define GPU_CAMBRICON_CNNL_CONV_FILTER_ADJUSTMENT_BASE_HPP

#include "cnnl.h"

#include "common/type_helpers.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"
#include "gpu/cambricon/sycl_bang_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

// TODO: this should really be fixed, cnnl and cudnn are quite different
struct cnnl_conv_filter_adjustment_base_t {
public:
    float filter_alpha_ = 1, filter_beta_ = 0;
    cnnlTensorDescriptor_t current_filter_desc_, transform_filter_desc_;
    // for filter in convolution, cnnl only support HWCN(4)、NHWC(4) and NDHWC(5).
    virtual bool supported_filter_format(const memory_desc_t *md) const {
        const memory_desc_wrapper mem_wrapper(md);
        // cnnl convolution supported 3 weight tensor layout
        return (!(mem_wrapper.matches_one_of_tag(format_tag::odhwi,  // corresponding to CNNL_LAYOUT_NDHWC for 5 dim conv
                    format_tag::hwio,   // corresponding to CNNL_LAYOUT_HWCN
                    format_tag::nhwc,   // corresponding to CNNL_LAYOUT_NHWC
                    format_tag::ohwi    // infact, it is still CNNL_LAYOUT_NHWC
                )));        
    }

    virtual ~cnnl_conv_filter_adjustment_base_t() {
        if (current_filter_desc_) {
            CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, current_filter_desc_);
        }
        if (transform_filter_desc_) {
            CNNL_EXECUTE_FUNC_V(cnnlDestroyTensorDescriptor, transform_filter_desc_);
        }
    }

    // ???
    void propagate_strides(int *strides, const int *dims,
            std::initializer_list<int> perm) const {
        int prev_p = -1;
        for (auto p : perm) {
            strides[p] = prev_p == -1 ? 1 : strides[prev_p] * dims[prev_p];
            prev_p = p;
        }
    }

    virtual status_t init_filter_transformation(
            cnnlDataType_t filter_data_types, int filter_ndims,
            int *filter_dims, int *current_filter_strides,
            int *transform_filter_strides) {
        // Set a descriptor for the current filter.
        // CHECK(create_and_set_tensor_descriptor(&current_filter_desc_,
        //         filter_data_types, layout, filter_ndims, filter_dims));
        // // Set a descriptor for the transform filter.
        // CHECK(create_and_set_tensor_descriptor(&transform_filter_desc_,
        //         filter_data_types, layout, filter_ndims, filter_dims));
        return status::unimplemented;
    }

    // TODO: this should be checked
    virtual void set_filter_hwcn(
            int filter_ndims, int *transform_filter_strides, int *filter_dims) {
        switch (filter_ndims) {
            case 4: // Convert to KCRS
                return propagate_strides(
                        transform_filter_strides, filter_dims, {1, 0, 2, 3});
            // case 6: not support
        }
    }

    virtual void set_filter_nhwc(
            int filter_ndims, int *transform_filter_strides, int *filter_dims) {
        switch (filter_ndims) {
            case 4: // Convert to krsc
                return propagate_strides(
                        transform_filter_strides, filter_dims, {1, 3, 2, 0});
            case 5:
                /// NOTE: Convert to krsdc. There is no support for krsdc and
                // 3d convolution in the current version. So we convert the
                // filter to ndhwc and then fold the dhwc for both srd and
                // filter to make it a 4d conv. So according to cnnl code
                // should looks like:
                // propagate_strides(
                //        transform_filter_strides, filter_dims, {1, 2, 4, 3,
                //        0});
                // However, executing the code shows that they actually expect
                // the filter format to be kdrsc.  Therefore, we convert the
                // filter to kdrsc:
                // propagate_strides(
                //      transform_filter_strides, filter_dims, {1, 4, 3, 2, 0});

                return propagate_strides(
                        transform_filter_strides, filter_dims, {1, 4, 3, 2, 0});
            // case 6: not support
        }
    }

    void set_filter_format(int filter_ndims, int *filter_dims,
            int *transform_filter_strides, cnnlTensorLayout_t format) {
        if (format == CNNL_LAYOUT_HWCN) {
            assert(0);  // TODO:
        } else {
            set_filter_nhwc(filter_ndims, transform_filter_strides, filter_dims);
        }
    }

    void transform_filter(cnnlHandle_t handle, void *current_filter,
            void *transform_filter) const {
        // CNNL_EXECUTE_FUNC(cnnlCreateTransposeDescriptor, &Transdesc);
        // CNNL_EXECUTE_FUNC(cnnlSetTransposeDescriptor, 4, {});    // this TDDO:
        // CNNL_EXECUTE_FUNC(cnnlTranspose, handle, current_filter_desc_, current_filter, transform_filter_desc_, transform_filter);
        // CNNL_EXECUTE_FUNC(cnnlDestroyTransposeDescriptor, &Transdesc);
    }

    void undo_transform_filter(cnnlHandle_t handle, void *transform_filter,
            void *current_filter) const {
        // TODO:
        // CNNL_EXECUTE_FUNC(cnnlCreateTransposeDescriptor, &Transdesc);
        // CNNL_EXECUTE_FUNC(cnnlSetTransposeDescriptor, 4, {});    // this TDDO:
        // CNNL_EXECUTE_FUNC(cnnlTranspose, handle, transform_filter_desc_, transform_filter current_filter_desc_, current_filter);
        // CNNL_EXECUTE_FUNC(cnnlDestroyTransposeDescriptor, &Transdesc);
    }
};

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
