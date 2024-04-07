/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_OCL_XE_HPC_CONVOLUTION_HPP
#define GPU_OCL_XE_HPC_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct xe_hpc_convolution_fwd_t : public gpu_primitive_t {
    struct pd_t : public gpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:xe_hpc", xe_hpc_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);

            const auto attr_skip_mask
                    = primitive_attr_t::skip_mask_t::oscale_runtime
                    | primitive_attr_t::skip_mask_t::post_ops
                    | primitive_attr_t::skip_mask_t::sum_dt;

            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            is_xe_hpc = compute_engine->is_xe_hpc();

            bool ok = is_xe_hpc
                    && utils::one_of(this->desc()->prop_kind, forward_training,
                            forward_inference)
                    && this->desc()->alg_kind == alg_kind::convolution_direct
                    && utils::one_of(true,
                            expect_data_types(f16, f16, f16, f16, f16),
                            expect_data_types(bf16, bf16, bf16, bf16, f32),
                            expect_data_types(bf16, bf16, f32, bf16, f32),
                            expect_data_types(bf16, bf16, bf16, f32, f32),
                            expect_data_types(bf16, bf16, f32, f32, f32),
                            expect_data_types(u8, s8, f32, u8, s32),
                            expect_data_types(u8, s8, f32, s8, s32),
                            expect_data_types(s8, s8, f32, u8, s32),
                            expect_data_types(s8, s8, f32, s8, s32))
                    && attr()->has_default_values(
                            attr_skip_mask, desc()->dst_desc.data_type)
                    && post_ops_with_binary_ok(
                            attr(), desc()->dst_desc.data_type)
                    && IMPLICATION(!attr()->output_scales_.has_default_values(),
                            utils::one_of(src_md_.data_type, s8, u8)
                                    && utils::one_of(
                                            attr()->output_scales_.mask_, 0,
                                            1 << 1));
            if (!ok) return status::unimplemented;

            CHECK(init_conf());

            if (!compute_engine->mayiuse_sub_group(conf.sub_group_size))
                return status::unimplemented;

            CHECK(init_scales_md());

            ok = set_default_formats_common(
                    conf.src_tag, conf.wei_tag, conf.dst_tag);
            if (!ok) return status::unimplemented;

            CHECK(attr_.set_default_formats(dst_md(0)));

            return status::success;
        }

        bool is_xe_hpc = false;
        status_t init_conf();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        const memory_desc_t *scales_md() const { return &scales_md_; }

        conv_conf_t conf;

    private:
        status_t init_scales_md() {
            if (!conf.attr_info.with_per_oc_oscales) return status::success;

            scales_md_.data_type = data_type::f32;
            scales_md_.ndims = 1;
            scales_md_.dims[0] = attr()->output_scales_.count_;
            return memory_desc_init_by_tag(scales_md_, format_tag::x);
        }

        memory_desc_t scales_md_;
    };

    status_t init(engine_t *engine) override {
        const char *kernel_name = "xe_hpc_conv_fwd";

        compute::kernel_ctx_t kernel_ctx;
        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        create_kernel(engine, &kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    xe_hpc_convolution_fwd_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

protected:
    status_t init_res_storage(
            engine_t *engine, gpu_resource_t *r) const override {
        if (!pd()->conf.attr_info.with_per_oc_oscales
                || pd()->conf.attr_info.with_runtime_oscales)
            return status::success;

        memory_desc_wrapper scales_mdw(pd()->scales_md());
        auto scales_sz = scales_mdw.nelems() * sizeof(float);
        memory_storage_t *tmp_mem_storage_ptr;
        CHECK(engine->create_memory_storage(&tmp_mem_storage_ptr, scales_sz));

        std::unique_ptr<memory_storage_t> tmp_mem_storage(tmp_mem_storage_ptr);
        void *scales_ptr = nullptr;
        CHECK(tmp_mem_storage->map_data(&scales_ptr, nullptr, scales_sz));
        utils::array_copy((float *)scales_ptr,
                pd()->attr()->output_scales_.scales_,
                pd()->attr()->output_scales_.count_);
        CHECK(tmp_mem_storage->unmap_data(scales_ptr, nullptr));
        r->add_memory_storage(SCALES_, std::move(tmp_mem_storage));
        return status::success;
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)gpu_primitive_t::pd().get(); }

    compute::kernel_t kernel_;
    enum { SCALES_ = 0 };
};

struct xe_hpc_convolution_bwd_data_t : public gpu_primitive_t {
    struct pd_t : public gpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:xe_hpc", xe_hpc_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::oscale
                    | primitive_attr_t::skip_mask_t::post_ops;

            bool ok = utils::one_of(true,
                              expect_data_types(u8, s8, f32, u8, s32),
                              expect_data_types(u8, s8, f32, s8, s32),
                              expect_data_types(s8, s8, f32, u8, s32),
                              expect_data_types(s8, s8, f32, s8, s32),
                              expect_data_types(f16, f16, f16, f16, f16),
                              expect_data_types(bf16, bf16, bf16, bf16, f32))
                    && desc()->prop_kind == prop_kind::backward_data
                    && desc()->alg_kind == alg_kind::convolution_direct
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && attr()->has_default_values(attr_skip_mask)
                    && post_ops_ok(attr());

            if (!ok) return status::unimplemented;

            CHECK(init_conf());

            if (!compute_engine->mayiuse_sub_group(conf.sub_group_size))
                return status::unimplemented;

            ok = set_default_formats_common(
                    conf.src_tag, conf.wei_tag, conf.dst_tag);

            is_xe_hpc = compute_engine->is_xe_hpc();
            return ok && is_xe_hpc ? status::success : status::unimplemented;
        }

        status_t init_conf();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        bool is_xe_hpc = false;
        conv_conf_t conf;

        bool support_bias() const override { return true; }
    };

    status_t init(engine_t *engine) override {
        const char *kernel_name = "xe_hpc_conv_bwd_data";
        compute::kernel_ctx_t kernel_ctx;
        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        create_kernel(engine, &kernel_, kernel_name, kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    xe_hpc_convolution_bwd_data_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)gpu_primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
