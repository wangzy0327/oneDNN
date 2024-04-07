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

#include "gpu/jit/conv/gen_convolution.hpp"

#include <iostream>
#include <utility>

#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/conv_kernel.hpp"
#include "gpu/jit/conv/kernel_info.hpp"
#include "gpu/jit/conv/utils.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace compute;

static size_t icache_size(ngen::HW arch) {
    switch (arch) {
        case gpu_gen9: return 48 * 1024;
        case gpu_xe_lp: return 48 * 1024;
        case gpu_xe_hp: return 48 * 1024;
        case gpu_xe_hpg: return 96 * 1024;
        case gpu_xe_hpc: return 80 * 1024;
        default: return 0;
    }
}

template <template <ngen::HW> class KernelT, ngen::HW arch, typename... ArgsT>
std::unique_ptr<jit::jit_generator_base> make_generator(ArgsT &&... args) {

    auto raw_kernel = new KernelT<arch>(std::forward<ArgsT>(args)...);
    if (raw_kernel->getRootStreamLength() > icache_size(arch)) {
        ir_warning() << raw_kernel->kernel_name()
                     << " larger than icache, kernel: "
                     << raw_kernel->getRootStreamLength()
                     << " bytes, icache: " << icache_size(arch) << " bytes\n";
    }
    return std::unique_ptr<jit::jit_generator_base>(raw_kernel);
}

template <template <ngen::HW> class KernelT, typename... ArgsT>
compute::kernel_t make_kernel(
        gpu_primitive_t *primitive, engine_t *engine, ArgsT &&... args) {
    auto compute_engine = utils::downcast<compute_engine_t *>(engine);
    auto device_info = compute_engine->device_info();

    std::unique_ptr<jit::jit_generator_base> jit_kernel;
    switch (device_info->gpu_arch()) {
        case gpu_arch_t::gen9:
            jit_kernel = make_generator<KernelT, gpu_gen9>(
                    std::forward<ArgsT>(args)...);
            break;
        case gpu_arch_t::xe_lp:
            jit_kernel = make_generator<KernelT, gpu_xe_lp>(
                    std::forward<ArgsT>(args)...);
            break;
        case gpu_arch_t::xe_hp:
            jit_kernel = make_generator<KernelT, gpu_xe_hp>(
                    std::forward<ArgsT>(args)...);
            break;
        case gpu_arch_t::xe_hpg:
            jit_kernel = make_generator<KernelT, gpu_xe_hpg>(
                    std::forward<ArgsT>(args)...);
            break;
        case gpu_arch_t::xe_hpc:
            jit_kernel = make_generator<KernelT, gpu_xe_hpc>(
                    std::forward<ArgsT>(args)...);
            break;
        default: break;
    }

    if (!jit_kernel) return compute::kernel_t();

    compute::kernel_t kernel;
    status_t status = primitive->create_kernel(engine, &kernel, *jit_kernel);
    if (status != status::success) return compute::kernel_t();
    return kernel;
}

class gen_convolution_t {
public:
    template <typename T>
    static status_t init_pd(T *pd, engine_t *engine) {
        auto *compute_engine = utils::downcast<compute_engine_t *>(engine);

        if (!compute_engine->mayiuse_ngen_kernels())
            return status::unimplemented;
        if (!pd->set_default_alg_kind(alg_kind::convolution_direct))
            return status::unimplemented;
        pd->cfg = std::make_shared<conv_config_t>();
        CHECK(pd->cfg->init(pd, &pd->attr_, engine));
        CHECK(init_kernel_infos(pd));

        return status::success;
    }

    gen_convolution_t() = default;

    template <typename T>
    status_t init(T *primitive, engine_t *engine) {
        try {
            auto &cfg = get_cfg(primitive);

            ir_trace() << "Configuration:" << std::endl;
            ir_trace() << cfg;

            auto &kernel_infos = primitive->pd()->kernel_infos;
            for (int i = 0; i < int(kernel_infos.size()); i++) {
                auto &info = *kernel_infos[i];
                switch (info.id()) {
                    case kernel_id_t::convolution:
                        kernels_.push_back(make_kernel<conv_kernel_t>(
                                primitive, engine, cfg, primitive->pd(), info));
                        break;
                    case kernel_id_t::pre_reorder: {
                        auto src_layout = cfg.tensor_config.user_layout(
                                info.arg_name(1));
                        auto dst_layout = cfg.tensor_config.compute_layout(
                                info.arg_name(1));
                        kernels_.push_back(make_kernel<reorder_kernel_t>(
                                primitive, engine, cfg, primitive->pd(), info,
                                src_layout, dst_layout));
                        break;
                    }
                    case kernel_id_t::post_reorder: {
                        auto src_layout = cfg.tensor_config.compute_layout(
                                info.arg_name(0));
                        auto dst_layout = cfg.tensor_config.user_layout(
                                info.arg_name(0));
                        kernels_.push_back(make_kernel<reorder_kernel_t>(
                                primitive, engine, cfg, primitive->pd(), info,
                                src_layout, dst_layout));
                        break;
                    }
                    case kernel_id_t::zero_out:
                        kernels_.push_back(make_kernel<zero_out_kernel_t>(
                                primitive, engine, cfg, primitive->pd(), info));
                        break;
                    default: ir_error_not_expected();
                }
                if (!kernels_[i]) return status::runtime_error;
            }
        } catch (...) {
            // If verbose is enabled, print the primitive case and rethrow the
            // exception.
            if (get_verbose())
                printf("onednn_verbose,error,%s\n",
                        primitive->pd()->info(engine));
            throw;
        }

        return status::success;
    }

    template <typename T>
    status_t init_res_storage(
            const T *primitive, engine_t *engine, gpu_resource_t *r) const {
        auto &kernel_infos = primitive->pd()->kernel_infos;
        for (int i = 0; i < int(kernel_infos.size()); i++) {
            auto &kernel_info = *kernel_infos[i];
            for (int j = 0; j < kernel_info.nargs(); j++) {
                if (!kernel_info.is_resource(j)) continue;

                auto &arg_name = kernel_info.arg_name(j);
                int key = kernel_info.key(j);
                if (arg_name == "oscales") {
                    CHECK(primitive->init_output_scales_res_storage(
                            engine, r, key));
                } else {
                    ir_error_not_expected();
                }
            }
        }
        return status::success;
    }

    template <typename T>
    status_t execute(const T *primitive, const exec_ctx_t &ctx) const {
        auto &kernel_infos = primitive->pd()->kernel_infos;

        int max_stage = 100;
        int nsubmitted = 0;
        int nkernels = int(kernel_infos.size());
        for (int stage = 0; stage < max_stage; stage++) {
            for (int i = 0; i < nkernels; i++) {
                auto &info = *kernel_infos[i];
                if (info.stage_id() != stage) continue;

                std::vector<memory_storage_wrapper_t> storage_list;
                info.init_memory_storage_list(storage_list, ctx, primitive);

                kernel_arg_list_t arg_list;
                info.set_args(arg_list, storage_list);

                CHECK(primitive->parallel_for(
                        ctx, info.nd_range(), kernels_[i], arg_list));
                nsubmitted++;
                if (nsubmitted == nkernels) break;
            }
        }

        return status::success;
    }

private:
    template <typename T>
    static const conv_config_t &get_cfg(const T *primitive) {
        return *primitive->pd()->cfg;
    }

    template <typename T>
    static kernel_info_t &create_kernel_info(T *pd, kernel_id_t kernel_id) {
        auto &infos = pd->kernel_infos;
        infos.push_back(std::make_shared<kernel_info_t>());
        auto &ret = *infos.back();
        ret.set_id(kernel_id);
        return ret;
    }

    template <typename T>
    static status_t init_kernel_infos(T *pd) {
        auto &cfg = *pd->cfg;
        auto *attr = pd->attr();

        auto scratchpad = pd->scratchpad_registry().registrar();

        auto &conv_info = create_kernel_info(pd, kernel_id_t::convolution);

        // Initialize kernel arguments.
        uint32_t scratchpad_key = 1;
        for (auto &t : cfg.tensor_config.tensors()) {
            int compute_arg_key = t.arg_key;
            int user_arg_key = t.arg_key;
            size_t elems = t.compute_layout.elems();
            size_t compute_size = t.compute_layout.size();
            auto compute_buf = make_buffer(t.name);
            auto user_buf = (t.needs_reorder ? make_buffer(t.name + "_user")
                                             : compute_buf);

            if (user_arg_key == -1) {
                ir_assert(!t.needs_reorder);
                ir_assert(!t.needs_zero_out);

                if (t.name == "oscales") {
                    if (elems == 1) {
                        auto oscales_var = var_t::make(type_t::f32(), t.name);
                        conv_info.register_internal_arg(
                                oscales_var, attr->output_scales_.scales_[0]);
                    } else {
                        conv_info.register_resource_arg(make_buffer(t.name));
                    }
                } else {
                    ir_error_not_expected();
                }
                continue;
            }

            if (t.needs_reorder) {
                compute_arg_key = int(scratchpad_key);
                scratchpad.book(scratchpad_key, compute_size, 1,
                        ocl::OCL_BUFFER_ALIGNMENT);
                conv_info.register_scratchpad_arg(compute_buf, compute_arg_key,
                        /*is_input=*/t.is_input && !t.is_output, compute_size);
                scratchpad_key++;

                if (t.is_input) {
                    auto &reorder_info
                            = create_kernel_info(pd, kernel_id_t::pre_reorder);
                    reorder_info.register_user_arg(user_buf, user_arg_key,
                            /*is_input=*/true);
                    reorder_info.register_scratchpad_arg(compute_buf,
                            compute_arg_key,
                            /*is_input=*/false, compute_size);
                    auto elems_var = var_t::make(type_t::u32(), "elems");
                    reorder_info.register_internal_arg(
                            elems_var, uint32_t(elems));
                    reorder_info.set_nd_range(reorder_kernel_t<>::nd_range(
                            cfg.simd_size, t.user_layout, t.compute_layout));
                }
                if (t.is_output) {
                    auto &reorder_info
                            = create_kernel_info(pd, kernel_id_t::post_reorder);
                    reorder_info.register_scratchpad_arg(compute_buf,
                            compute_arg_key,
                            /*is_input=*/true, compute_size);
                    reorder_info.register_user_arg(user_buf, user_arg_key,
                            /*is_input=*/false);
                    auto elems_var = var_t::make(type_t::u32(), "elems");
                    reorder_info.register_internal_arg(
                            elems_var, uint32_t(elems));
                    reorder_info.set_nd_range(reorder_kernel_t<>::nd_range(
                            cfg.simd_size, t.compute_layout, t.user_layout));
                }
            }
            if (t.needs_zero_out) {
                auto &zero_out_info
                        = create_kernel_info(pd, kernel_id_t::zero_out);
                if (t.needs_reorder) {
                    zero_out_info.register_scratchpad_arg(compute_buf,
                            compute_arg_key,
                            /*is_input=*/false, compute_size);
                } else {
                    zero_out_info.register_user_arg(compute_buf,
                            compute_arg_key,
                            /*is_input=*/false);
                }
                auto size_var = var_t::make(type_t::u32(), "size");
                zero_out_info.register_internal_arg(
                        size_var, uint32_t(compute_size));
                int bytes_per_thr = zero_out_kernel_t<>::bytes_per_thr;
                compute::nd_range_t nd_range(
                        {utils::div_up(compute_size, bytes_per_thr)
                                * cfg.simd_size});
                zero_out_info.set_nd_range(nd_range);
            }
            if (!t.needs_reorder)
                conv_info.register_user_arg(user_buf, user_arg_key,
                        /*is_input=*/t.is_input && !t.is_output);
        }

        conv_info.set_nd_range(cfg.nd_range());

        return status::success;
    }

    std::vector<kernel_t> kernels_;
};

status_t gen_convolution_fwd_t::pd_t::init(engine_t *engine) {
    if (!is_fwd()) return status::unimplemented;
    CHECK(gen_convolution_t::init_pd(this, engine));
    return status::success;
}

status_t gen_convolution_fwd_t::init(engine_t *engine) {
    impl_.reset(new gen_convolution_t());
    return impl_->init(this, engine);
}

status_t gen_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

status_t gen_convolution_fwd_t::init_res_storage(
        engine_t *engine, gpu_resource_t *r) const {
    return impl_->init_res_storage(this, engine, r);
}

status_t gen_convolution_bwd_data_t::pd_t::init(engine_t *engine) {
    if (!is_bwd_d()) return status::unimplemented;
    CHECK(gen_convolution_t::init_pd(this, engine));
    return status::success;
}

status_t gen_convolution_bwd_data_t::init_res_storage(
        engine_t *engine, gpu_resource_t *r) const {
    return impl_->init_res_storage(this, engine, r);
}

status_t gen_convolution_bwd_weights_t::pd_t::init(engine_t *engine) {
    if (!is_bwd_w()) return status::unimplemented;
    CHECK(gen_convolution_t::init_pd(this, engine));
    return status::success;
}

status_t gen_convolution_bwd_data_t::init(engine_t *engine) {
    impl_.reset(new gen_convolution_t());
    return impl_->init(this, engine);
}

status_t gen_convolution_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

status_t gen_convolution_bwd_weights_t::init(engine_t *engine) {
    impl_.reset(new gen_convolution_t());
    return impl_->init(this, engine);
}

status_t gen_convolution_bwd_weights_t::init_res_storage(
        engine_t *engine, gpu_resource_t *r) const {
    return impl_->init_res_storage(this, engine, r);
}

status_t gen_convolution_bwd_weights_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
