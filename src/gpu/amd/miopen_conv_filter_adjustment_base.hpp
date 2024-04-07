#ifndef GPU_AMD_MIOPEN_CONV_FILTER_ADJUSTMENT_BASE_HPP
#define GPU_AMD_MIOPEN_CONV_FILTER_ADJUSTMENT_BASE_HPP

#include "rocblas.h"
#include "miopen/miopen.h"

#include "common/type_helpers.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_conv_filter_adjustment_base_t {
public:
    float filter_alpha_ = 1, filter_beta_ = 0;
    miopenTensorDescriptor_t current_filter_desc_, transform_filter_desc_;
    // for filter in convolution, miopen only support nchw.
    // unsupported format like nhwc hwio and dhwio should be converted to nchw format. 
    virtual bool supported_filter_format(const memory_desc_t *md) const {
        const memory_desc_wrapper mem_wrapper(md);
        /// NOTE: the transformation for oidhw to oihwd is disabled until cuDNN
        // fixes the current bug for oihwd format. the transformation for
        // odhwi to ohwdi has been disabled until cuDNN provides support for
        // 3d convolution in ohwdi format.
        return (!(mem_wrapper.matches_one_of_tag(/*format_tag::oidhw,*/
                /*format_tag::odhwi,*/ format_tag::dhwio, format_tag::hwio)));
    }

    virtual ~miopen_conv_filter_adjustment_base_t() {
        if (current_filter_desc_) {
            MIOPEN_EXECUTE_FUNC_V(
                    miopenDestroyTensorDescriptor, current_filter_desc_);
        }
        if (transform_filter_desc_) {
            MIOPEN_EXECUTE_FUNC_V(
                    miopenDestroyTensorDescriptor, transform_filter_desc_);
        }
    }

    void propagate_strides(int *strides, const int *dims,
            std::initializer_list<int> perm) const {
        int prev_p = -1;
        for (auto p : perm) {
            strides[p] = prev_p == -1 ? 1 : strides[prev_p] * dims[prev_p];
            prev_p = p;
        }
    }

    virtual status_t init_filter_transformation(
            miopenDataType_t filter_data_types, int filter_ndims,
            int *filter_dims, int *current_filter_strides,
            int *transform_filter_strides) {
        // Set a descriptor for the current filter.
        CHECK(create_and_set_tensor_descriptor(&current_filter_desc_,
                filter_data_types, filter_ndims, filter_dims,
                current_filter_strides));
        // Set a descriptor for the transform filter.
        CHECK(create_and_set_tensor_descriptor(&transform_filter_desc_,
                filter_data_types, filter_ndims, filter_dims,
                transform_filter_strides));
        return status::success;
    }

    virtual void set_filter_nchw(
            int filter_ndims, int *transform_filter_strides, int *filter_dims) {
        switch (filter_ndims) {
            case 4: // Convert to KCRS
                return propagate_strides(
                        transform_filter_strides, filter_dims, {3, 2, 1, 0});
            case 5:
                /// NOTE: cuDNN claims the filter must be in kcrsd . However
                // in the current version(7.6.5) it accepts kcdrs filter is the
                // same as ncdhw tensor. So according to cuDNN code should
                // looks like:
                // propagate_strides(
                //      transform_filter_strides, filter_dims, {2, 4, 3, 1, 0});
                // However, executing the code shows that they actually expect
                // the filter format to be kcdrs. Therefore, we convert the
                // filter to kcdrs instead:
                // propagate_strides(
                //      transform_filter_strides, filter_dims, {4, 3, 2, 1, 0});

                return propagate_strides(
                        transform_filter_strides, filter_dims, {4, 3, 2, 1, 0});
            case 6:
                return propagate_strides(transform_filter_strides, filter_dims,
                        {5, 4, 3, 2, 1, 0});
        }
    }

    void set_filter_format(int filter_ndims, int *filter_dims,
            int *transform_filter_strides) {
        set_filter_nchw(filter_ndims, transform_filter_strides, filter_dims);
    }

    void transform_filter(miopenHandle_t handle, void *current_filter,
            void *transform_filter) const {
        MIOPEN_EXECUTE_FUNC(miopenTransformTensor, handle, &filter_alpha_,
                current_filter_desc_, current_filter, &filter_beta_,
                transform_filter_desc_, transform_filter);
    }
    void undo_transform_filter(miopenHandle_t handle, void *transform_filter,
            void *current_filter) const {
        MIOPEN_EXECUTE_FUNC(miopenTransformTensor, handle, &filter_alpha_,
                transform_filter_desc_, transform_filter, &filter_beta_,
                current_filter_desc_, current_filter);
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
