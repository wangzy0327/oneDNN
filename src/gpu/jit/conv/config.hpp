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

#ifndef GPU_JIT_CONV_CONFIG_HPP
#define GPU_JIT_CONV_CONFIG_HPP

#include <iostream>
#include <sstream>

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/math_utils.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/compute/compute_engine.hpp"
#include "gpu/jit/conv/fma_support.hpp"
#include "gpu/jit/conv/tensor.hpp"
#include "gpu/jit/conv/tensor_config.hpp"
#include "gpu/jit/conv/utils.hpp"
#include "gpu/jit/jit_eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Description of the convolution problem.
class conv_problem_t {
public:
    conv_problem_t() = default;

    status_t init(convolution_pd_t *conv_pd) {
        if (conv_pd->has_zero_dim_memory()) return status::unimplemented;

        is_fwd = conv_pd->is_fwd();
        is_bwd_d = conv_pd->is_bwd_d();
        is_bwd_w = conv_pd->is_bwd_w();
        with_bias = conv_pd->with_bias();
        with_groups = conv_pd->with_groups();

        src_data_type = conv_pd->invariant_src_md()->data_type;
        wei_data_type = conv_pd->invariant_wei_md()->data_type;
        bia_data_type = conv_pd->invariant_bia_md()->data_type;
        dst_data_type = conv_pd->invariant_dst_md()->data_type;

        ndims = conv_pd->ndims();

        mb = conv_pd->MB();
        g = conv_pd->G();
        ic = ir_utils::safe_divide(conv_pd->IC(), g);
        oc = ir_utils::safe_divide(conv_pd->OC(), g);

        // Input spatial.
        id = conv_pd->ID();
        ih = conv_pd->IH();
        iw = conv_pd->IW();

        // Output spatial.
        od = conv_pd->OD();
        oh = conv_pd->OH();
        ow = conv_pd->OW();

        // Kernel sizes.
        kd = conv_pd->KD();
        kh = conv_pd->KH();
        kw = conv_pd->KW();

        // Strides.
        sd = conv_pd->KSD();
        sh = conv_pd->KSH();
        sw = conv_pd->KSW();

        // Padding.
        pd = conv_pd->padFront();
        ph = conv_pd->padT();
        pw = conv_pd->padL();

        // Dilation.
        dd = conv_pd->KDD();
        dh = conv_pd->KDH();
        dw = conv_pd->KDW();

        try_reduce_to_1d();

        is_dw = with_groups && (g > 1) && (oc == 1) && (ic == 1);

        return status::success;
    }

    bool is_stride1() const { return sd == 1 && sh == 1 && sw == 1; }

    // Reduces dimensions for 1x1 kernel.
    void try_reduce_to_1d() {
        bool is_1x1 = (kd * kh * kw == 1);
        bool is_eq_oi = (od == id && oh == ih && ow == iw);
        if (is_1x1 && is_stride1() && is_eq_oi) {
            ir_assert(pd == 0 && ph == 0 && pw == 0);
            ow = od * oh * ow;
            iw = id * ih * iw;
            od = id = kd = 1;
            oh = ih = kh = 1;
            reduced_to_1d = true;
        }
    }

    std::string desc_str() const {
        std::ostringstream oss;
        oss << "mb" << mb;
        oss << "g" << g;
        oss << "ic" << ic;
        oss << "id" << id;
        oss << "ih" << ih;
        oss << "iw" << iw;
        oss << "oc" << oc;
        oss << "od" << od;
        oss << "oh" << oh;
        oss << "ow" << ow;
        oss << "kd" << kd;
        oss << "kh" << kh;
        oss << "kw" << kw;
        if (sd != 1) oss << "sd" << sd;
        if (sh != 1) oss << "sh" << sh;
        if (sw != 1) oss << "sw" << sw;
        if (dd != 0) oss << "dd" << dd;
        if (dh != 0) oss << "dh" << dh;
        if (dw != 0) oss << "dw" << dw;
        oss << "pd" << pd;
        oss << "ph" << ph;
        oss << "pw" << pw;
        return oss.str();
    }

    tensor_config_t tensor_config;

    data_type_t src_data_type;
    data_type_t wei_data_type;
    data_type_t dst_data_type;
    data_type_t bia_data_type;

    bool is_fwd;
    bool is_bwd_d;
    bool is_bwd_w;
    bool with_bias;
    bool with_groups;
    bool is_dw;

    int ndims;
    int mb; // Batch size.
    int g; // Groups.
    int ic, oc; // Input and output channels.
    int id, ih, iw; // Input spatial sizes.
    int od, oh, ow; // Output spatial sizes.
    int kd, kh, kw; // Kernel sizes.
    int sd, sh, sw; // Strides.
    int pd, ph, pw; // Padding in the beginning.
    int dd, dh, dw; // Dilation.
    bool reduced_to_1d; // Whether the problem spatial was reduced to 1D.
};

// Parameters for kernel generation.
class conv_config_t : public conv_problem_t {
public:
    conv_config_t() = default;

    status_t init(convolution_pd_t *conv_pd, primitive_attr_t *attr,
            engine_t *engine) {
        // These functions have implicit dependencies between them. They cannot be
        // reordered with verifying these dependencies are satisfied.
        CHECK(conv_problem_t::init(conv_pd));
        CHECK(init_hw(engine));
        CHECK(init_abc_data_types());
        CHECK(init_acc_data_type());
        CHECK(init_fma_kind());
        CHECK(init_data_layouts(conv_pd));

        if (!data_types_ok()) return status::unimplemented;

        // Group convolution is not supported.
        // Depthwise convolution is supported for forward.
        if (with_groups && g > 1 && !(is_dw && is_fwd))
            return status::unimplemented;

        CHECK(init_common_config());

        const memory_desc_t *output_md = nullptr;
        if (is_fwd) {
            CHECK(init_fwd(conv_pd, engine));
            output_md = conv_pd->dst_md();
        } else if (is_bwd_d) {
            CHECK(init_bwd_d(conv_pd));
            output_md = conv_pd->diff_src_md();
        } else if (is_bwd_w) {
            CHECK(init_bwd_w(conv_pd));
            output_md = conv_pd->diff_weights_md();
        } else {
            ir_error_not_expected();
        }

        CHECK(attr->set_default_formats(output_md));

        if (!post_ops_ok(conv_pd)) return status::unimplemented;
        if (!hw_ok(engine)) return status::unimplemented;

        CHECK(init_extra_tensor_layouts(conv_pd));

        return status::success;
    }

    status_t init_fwd(convolution_pd_t *conv_pd, engine_t *engine) {
        using namespace ir_utils;

        if (ic < 16 && !is_dpas_fma() && !is_dw) return status::unimplemented;

        bool is_src_nhwc = is_nhwc("src", conv_pd);
        bool is_dst_nhwc = is_nhwc("dst", conv_pd);

        // Set dispatch and kernel parameters.
        if (is_dw) {
            g_tg_blk = (is_int8_dst() ? 32 : 16);
            mb_thr_blk
                    = (mb < 16 || is_src_nhwc ? 1
                                              : hw <= ngen::HW::XeLP ? 8 : 16);
            mb_thr_dim = (mb_thr_blk == 1 ? 1 : 2);
            ow_thr_blk = (mb_thr_blk == 1 ? 8 : 1);
            ow_thr_dim = 1;
            oc_thr_blk = 1;
            oc_thr_dim = 1;
            ic_thr_dim = 1;
            ic_blk = 1;

            int iw_load_blk = (ow_thr_blk - 1) * sw + (kw - 1) + 1;
            bool do_kw_buf = (kw > 1 && mb_thr_blk == 1 && iw_load_blk <= 32);
            kw_blk = (do_kw_buf ? kw : 1);
        } else if (fma_kind == fma_kind_t::mad
                && src_data_type == data_type::f32) {
            const int max_tg_size = 16;
            g_tg_blk = 1;
            mb_thr_blk = (mb < 16 ? 1 : 8);
            mb_thr_dim = std::min((mb_thr_blk != 1) ? (32 / mb_thr_blk) : 1,
                    utils::div_up(mb, mb_thr_blk));
#ifdef GEN_CONV_DEBUG
            mb_thr_blk = getenv_int("mb_thr_blk", mb_thr_blk);
#endif
            oc_thr_blk = 16;
            oc_thr_dim = std::min(4, utils::div_up(oc, oc_thr_blk));
            oc_thr_dim = (1 << math::ilog2q(oc_thr_dim));

            if (mb_thr_dim > 1) {
                ow_thr_blk = 1;
                ow_thr_dim = 1;
            } else {
                const int pref_ow_thr_dim
                        = max_tg_size / (oc_thr_dim * mb_thr_dim);
                const int pref_ow_block
                        = (mb_thr_blk == 1) ? 8 : kw > 1 ? 4 : 1;
                ow_thr_blk = ow < pref_ow_block * pref_ow_thr_dim
                        ? (1 << math::ilog2q(
                                   utils::div_up(ow, pref_ow_thr_dim)))
                        : pref_ow_block;
                ow_thr_dim = pref_ow_thr_dim;
            }
            ic_thr_dim = 1;
            kw_blk = 1;
            ic_blk = (is_small_ic() ? ic : 16);
        } else if (is_dpas_fma()) {
            g_tg_blk = 1;
            mb_thr_blk = is_small_ic() ? 8 : (mb < 16 ? 1 : 32);
            mb_thr_dim = (is_small_ic())
                    ? (mb < 16 ? std::min(utils::div_up(mb, mb_thr_blk), 4) : 4)
                    : 1;
            oc_thr_blk = 32;
            if (hw >= ngen::HW::XeHPC && !is_small_ic()) oc_thr_blk = 64;
            oc_thr_dim = std::min(4, utils::div_up(oc, oc_thr_blk));
            oc_thr_dim = (1 << math::ilog2q(oc_thr_dim));
            if (is_small_ic()) {
                ow_thr_blk = 4;
            } else {
                ow_thr_blk = (mb < 16 ? 16 : 1);
                if (ow < ow_thr_blk) ow_thr_blk = 8;
            }
            ow_thr_dim = is_small_ic()
                    ? 1
                    : std::min(4, utils::div_up(ow, ow_thr_blk));
            if (is_small_ic()) {
                kw_blk = 8;
                ic_blk = (is_s32_accumulator() ? 4 : 2);
            } else {
                kw_blk = 1;
                ic_blk = (is_s32_accumulator() ? 32 : 16);
            }

            ic_thr_dim = init_fwd_ic_thr_dim(
                    engine, mb_thr_blk, oc_thr_blk, ow_thr_blk, ic_blk);

            // Disable M/N thread group blocking when K thread group blocking
            // is enabled. For some reason combining them results in lower
            // performance.
            if (ic_thr_dim > 1) {
                ow_thr_dim = 1;
                oc_thr_dim = 1;
            }
        } else {
            ir_error_not_expected();
        }
        g_thr_blk = g_tg_blk;

        int ic_padded = utils::rnd_up(ic, ic_blk);
        ic_thr_blk = ir_utils::safe_divide(ic_padded, ic_thr_dim);

        ow_thr_dim = (1 << math::ilog2q(ow_thr_dim));

#ifdef GEN_CONV_DEBUG
        mb_thr_blk = getenv_int("mb_thr_blk", mb_thr_blk);
        mb_thr_dim = getenv_int("mb_thr_dim", mb_thr_dim);
        oc_thr_blk = getenv_int("oc_thr_blk", oc_thr_blk);
        oc_thr_dim = getenv_int("oc_thr_dim", oc_thr_dim);
        ow_thr_blk = getenv_int("ow_thr_blk", ow_thr_blk);
        ow_thr_dim = getenv_int("ow_thr_dim", ow_thr_dim);
#endif

        tg_grid_dim[0] = oc_thr_dim;
        tg_grid_dim[1] = mb_thr_dim * ow_thr_dim;
        tg_grid_dim[2] = ic_thr_dim;

        // Round down to a power of 2.
        tg_grid_dim[0] = (1 << math::ilog2q(tg_grid_dim[0]));
        tg_grid_dim[1] = (1 << math::ilog2q(tg_grid_dim[1]));
        tg_grid_dim[2] = (1 << math::ilog2q(tg_grid_dim[2]));

#ifdef GEN_CONV_DEBUG
        tg_grid_dim[0] = getenv_int("tg0", tg_grid_dim[0]);
        tg_grid_dim[1] = getenv_int("tg1", tg_grid_dim[1]);
#endif

        mb_tg_blk = mb_thr_dim * mb_thr_blk;
        oc_tg_blk = oc_thr_dim * oc_thr_blk;
        ow_tg_blk = ow_thr_dim * ow_thr_blk;

#ifdef GEN_CONV_DEBUG
        mb_tg_blk = getenv_int("mb_tg_blk", mb_tg_blk);
        oc_tg_blk = getenv_int("oc_tg_blk", oc_tg_blk);
        ow_tg_blk = getenv_int("ow_tg_blk", ow_tg_blk);
#endif

        // TODO: Update estimate_register_count.
        b_blk = g_tg_blk;
        m_tg_blk = mb_tg_blk * ow_tg_blk;
        n_tg_blk = oc_tg_blk;
        k_blk = ic_blk * kw_blk;

        int g_tg_padded = utils::rnd_up(g, g_tg_blk);
        int mb_tg_padded = utils::rnd_up(mb, mb_tg_blk);
        int oc_tg_padded = utils::rnd_up(oc, oc_tg_blk);
        int ow_tg_padded = utils::rnd_up(ow, ow_tg_blk);

        g_tg_dim = g_tg_padded / g_tg_blk;
        mb_tg_dim = mb_tg_padded / mb_tg_blk;
        oc_tg_dim = oc_tg_padded / oc_tg_blk;

        ow_tg_dim = ow_tg_padded / ow_tg_blk;

        kernel_grid_dim[0] = oc_tg_dim;
        kernel_grid_dim[1] = g_tg_dim * od * oh * ow_tg_dim;
        kernel_grid_dim[2] = mb_tg_dim;

        allow_grf_reorder = is_small_ic() || is_dw;

        if (kd * kh * kw > 9) do_loop_unroll = false;
        if (is_dw) {
            use_preload = false;
            do_loop_unroll = false;
        }
        if (is_small_ic()) {
            reuse_headers = true;
            do_loop_unroll = false;
        }

        regs = hw <= ngen::HW::XeLP ? 128 : 256;

        // XXX: in case of nhwc or small mb allow reorders on XeHPC
        // since A/B tile loads may be strided
        if (hw >= ngen::HW::XeHPC
                && (mb_thr_blk == 1 || is_src_nhwc || is_dst_nhwc))
            allow_grf_reorder = true;

        if (mb >= 16) {
            // Large batch performance is slightly behind for some cases.
            bool large_batch_ok = false;
            if (hw >= ngen::HW::XeHPC) large_batch_ok = true;
            if (is_src_nhwc) large_batch_ok = true;
            // TODO: Fix issues with mb zero padding
            if (is_small_ic() && mb % 16 == 0) large_batch_ok = true;
            if (!large_batch_ok) return status::unimplemented;
        }

        fixup_inference_consistency();
        if (!try_reduce_grf_usage()) return status::unimplemented;

        return status::success;
    }

    status_t init_bwd_d(convolution_pd_t *conv_pd) {
        using namespace ir_utils;

        // Set dispatch and kernel parameters.
        mb_thr_blk = (mb < 16 ? 1 : 32);
        ic_thr_blk = 32;
        if (hw >= ngen::HW::XeHPC) ic_thr_blk = 64;
        iw_thr_blk = (mb < 16 ? 16 : 1);
        if (iw < iw_thr_blk) iw_thr_blk = 8;

#ifdef GEN_CONV_DEBUG
        mb_thr_blk = getenv_int("mb_thr_blk", mb_thr_blk);
        ic_thr_blk = getenv_int("ic_thr_blk", ic_thr_blk);
        iw_thr_blk = getenv_int("iw_thr_blk", iw_thr_blk);
#endif

        // Try to enable special optimization for strided BWD_D convolution.
        if (can_optimize_strided_bwd_d()) optimize_strided = true;

        regs = 256;

        iw_thr_dim = std::min(4, utils::div_up(iw, iw_thr_blk));
        iw_thr_dim = (1 << math::ilog2q(iw_thr_dim));

        if (optimize_strided) {
            iw_thr_dim = ir_utils::max_divisor(iw / sw, {1, 2, 4});
        }

        tg_grid_dim[0] = std::min(4, utils::div_up(ic, ic_thr_blk));
        tg_grid_dim[1] = iw_thr_dim;
        tg_grid_dim[2] = 1;

        // Round down to a power of 2.
        tg_grid_dim[0] = (1 << math::ilog2q(tg_grid_dim[0]));
        tg_grid_dim[2] = (1 << math::ilog2q(tg_grid_dim[2]));

#ifdef GEN_CONV_DEBUG
        tg_grid_dim[0] = getenv_int("tg0", tg_grid_dim[0]);
        tg_grid_dim[1] = getenv_int("tg1", tg_grid_dim[1]);
#endif

        mb_tg_blk = mb_thr_blk;
        ic_tg_blk = tg_grid_dim[0] * ic_thr_blk;
        iw_tg_blk = tg_grid_dim[1] * iw_thr_blk;
        oc_blk = (is_s32_accumulator() ? 32 : 16);

#ifdef GEN_CONV_DEBUG
        mb_tg_blk = getenv_int("mb_tg_blk", mb_tg_blk);
        ic_tg_blk = getenv_int("ic_tg_blk", ic_tg_blk);
        iw_tg_blk = getenv_int("iw_tg_blk", iw_tg_blk);
#endif

        m_tg_blk = mb_tg_blk * iw_tg_blk;
        n_tg_blk = ic_tg_blk;
        k_blk = oc_blk;

        int mb_tg_padded = utils::rnd_up(mb, mb_tg_blk);
        int ic_tg_padded = utils::rnd_up(ic, ic_tg_blk);
        int iw_tg_padded = utils::rnd_up(iw, iw_tg_blk);

        mb_tg_dim = mb_tg_padded / mb_tg_blk;
        ic_tg_dim = ic_tg_padded / ic_tg_blk;

        iw_tg_dim = iw_tg_padded / iw_tg_blk;

        kernel_grid_dim[0] = ic_tg_dim;
        kernel_grid_dim[1] = id * ih * iw_tg_dim;
        kernel_grid_dim[2] = mb_tg_dim;

        allow_grf_reorder = false;

        // Do not perform full unrolling when there are too many inner
        // iterations.
        int kernel_limit = is_f32_conv() ? 4 : 9;
        if (kd * kh * kw > kernel_limit) do_loop_unroll = false;

        // Do not perform full unrolling with non-unit stride unless special
        // stride optimization is enabled. These cases have non-trivial
        // post-increment updates which result in unrolling all reduction loops
        // and exceeding the instruction cache.
        if (!is_stride1() && !optimize_strided) do_loop_unroll = false;

        fixup_inference_consistency();
        if (!try_reduce_grf_usage()) return status::unimplemented;

        // Validate layouts.
        bool is_src_nhwc = is_nhwc("src", conv_pd);
        bool is_dst_nhwc = is_nhwc("dst", conv_pd);

        // XXX: in case of nhwc or small mb allow reorders on XeHPC
        // since A/B tile loads may be strided
        if (hw >= ngen::HW::XeHPC
                && (mb_thr_blk == 1 || is_src_nhwc || is_dst_nhwc))
            allow_grf_reorder = true;

        if (hw < ngen::HW::XeHPC)
            // Blocked large batch performance is slightly behind.
            if (!is_src_nhwc && mb >= 16) return status::unimplemented;

        return status::success;
    }

    status_t init_bwd_w(convolution_pd_t *conv_pd) {
        using namespace ir_utils;

        if (fma_kind == fma_kind_t::mad) {
            // Performance for small ic and small mb is worse than ocl:ncsp
            // implementation
            if (is_small_ic() && mb < 16) return status::unimplemented;

            oc_thr_blk = simd_size;
            ic_thr_blk = (ic < simd_size ? utils::rnd_up_pow2(ic) : simd_size);
            kw_blk = utils::rnd_up_pow2(
                    std::min(utils::div_up(simd_size, ic_thr_blk), kw));
            mb_blk = mb < 16 ? 1 : 16;
            mb_tg_blk = mb_blk;
            ow_thr_blk = mb < 16 ? std::min(16, utils::rnd_up_pow2(ow)) : 1;
        } else if (is_dpas_fma()) {
            oc_thr_blk = (oc <= 16 ? 16 : 32);
            if (hw >= ngen::HW::XeHPC) oc_thr_blk = (oc <= 16 ? 16 : 64);
            // Value required due to blocking in dpas data format
            int min_ic_thr_blk = is_s32_accumulator() ? 4 : 2;
            ic_thr_blk = (ic <= 16
                            ? std::max(utils::rnd_up_pow2(ic), min_ic_thr_blk)
                            : mb < 16 ? 16 : 32);
            kw_blk = utils::rnd_up_pow2(
                    std::min(utils::div_up(16, ic_thr_blk), kw));

            mb_blk = mb < 16 ? 1 : 16;
            mb_tg_blk = (mb < 16 || mb <= mb_blk) ? mb_blk : 2 * mb_blk;
            ow_thr_blk = mb < 16 ? 16 : 1;
            // TODO: Investigate why insufficient registers even though m_tg_blk is
            // the same
            if (mb < 16 && kw > 8 && kw_blk >= 8) kw_blk = 4;
        } else {
            ir_error_not_expected();
        }

#ifdef GEN_CONV_DEBUG
        oc_thr_blk = getenv_int("oc_thr_blk", oc_thr_blk);
        ic_thr_blk = getenv_int("ic_thr_blk", ic_thr_blk);
        kw_blk = getenv_int("kw_blk", kw_blk);
        ow_thr_blk = getenv_int("ow_thr_blk", ow_thr_blk);
        mb_blk = getenv_int("mb_blk", mb_blk);
        mb_tg_blk = getenv_int("mb_tg_blk", mb_tg_blk);
#endif

        kw_tg_dim = utils::div_up(kw, kw_blk);

        int max_oc_thr_dim = 4;
        int max_ic_thr_dim = 4;

        // Prefer larger thread groups when possible on XeHPC.
        if (hw >= ngen::HW::XeHPC) {
            if (oc / oc_thr_blk >= 8) {
                max_oc_thr_dim = 8;
            } else {
                max_ic_thr_dim = 8;
            }
        }

        regs = 256;
        tg_grid_dim[0]
                = std::min(max_oc_thr_dim, utils::div_up(oc, oc_thr_blk));
        tg_grid_dim[1]
                = std::min(max_ic_thr_dim, utils::div_up(ic, ic_thr_blk));
        tg_grid_dim[2] = 1;

        // Round down to a power of 2.
        tg_grid_dim[0] = (1 << math::ilog2q(tg_grid_dim[0]));
        tg_grid_dim[1] = (1 << math::ilog2q(tg_grid_dim[1]));
        tg_grid_dim[2] = (1 << math::ilog2q(tg_grid_dim[2]));

#ifdef GEN_CONV_DEBUG
        tg_grid_dim[0] = getenv_int("tg0", tg_grid_dim[0]);
        tg_grid_dim[1] = getenv_int("tg1", tg_grid_dim[1]);
#endif

        oc_tg_blk = tg_grid_dim[0] * oc_thr_blk;
        ic_tg_blk = tg_grid_dim[1] * ic_thr_blk;
        kw_tg_blk = kw_blk;

        init_bwd_w_spatial_blocks();

        mb_unroll = mb_tg_blk / mb_blk;
        ow_unroll = mb < 16 && is_dpas_fma() ? ow_tg_blk / ow_thr_blk : 1;

        m_tg_blk = ic_tg_blk * kw_tg_blk;
        n_tg_blk = oc_tg_blk;
        k_blk = mb_blk * ow_thr_blk;

        int oc_tg_padded = utils::rnd_up(oc, oc_tg_blk);
        int ic_tg_padded = utils::rnd_up(ic, ic_tg_blk);
        int mb_tg_padded = utils::rnd_up(mb, mb_tg_blk);
        int od_tg_padded = utils::rnd_up(od, od_tg_blk);
        int oh_tg_padded = utils::rnd_up(oh, oh_tg_blk);
        int ow_tg_padded = utils::rnd_up(ow, ow_tg_blk);

        oc_tg_dim = oc_tg_padded / oc_tg_blk;
        ic_tg_dim = ic_tg_padded / ic_tg_blk;

        mb_tg_dim = mb_tg_padded / mb_tg_blk;
        od_tg_dim = od_tg_padded / od_tg_blk;
        oh_tg_dim = oh_tg_padded / oh_tg_blk;
        ow_tg_dim = ow_tg_padded / ow_tg_blk;

        kernel_grid_dim[0] = oc_tg_dim;
        kernel_grid_dim[1] = ic_tg_dim * kd * kh * kw_tg_dim * od_tg_dim
                * oh_tg_dim * ow_tg_dim;
        kernel_grid_dim[2] = mb_tg_dim;

        // Set BWD_W-specific settings.
        do_b_reduction = with_bias;
        do_loop_unroll = (hw >= ngen::HW::XeHPC && is_dpas_fma() && mb_blk > 1);
        allow_grf_reorder = is_dpas_fma();
        do_atomic_update = true;

        if (!with_sum_post_op(conv_pd)) {
            tensor_config.require_zero_out("wei");
            if (with_bias) tensor_config.require_zero_out("bia");
        }

        if (wei_data_type == data_type::bf16) {
            auto &bf16_layout = tensor_config.compute_layout("wei");
            tensor_config.set_compute_layout(
                    "wei", bf16_layout.retype(type_t::f32()));
        }

        if (bia_data_type == data_type::bf16) {
            auto &bf16_layout = tensor_config.compute_layout("bia");
            tensor_config.set_compute_layout(
                    "bia", bf16_layout.retype(type_t::f32()));
        }

#ifdef GEN_CONV_DEBUG
        do_loop_unroll = getenv_bool("do_loop_unroll", do_loop_unroll);
        allow_grf_reorder = getenv_bool("allow_grf_reorder", allow_grf_reorder);
#endif

        fixup_inference_consistency();
        if (!try_reduce_grf_usage()) return status::unimplemented;

        // XXX: disable f32 bwd_w due to hang
        if (hw == ngen::HW::XeHP || hw == ngen::HW::XeHPG)
            if (src_data_type == data_type::f32
                    && dst_data_type == data_type::f32)
                return status::unimplemented;

        return status::success;
    }

    void init_bwd_w_spatial_blocks() {
        od_tg_blk = 1;
        oh_tg_blk = 1;
        ow_tg_blk = ow_thr_blk;
        bool are_small_large_channels
                = (std::min(ic, oc) <= 64 && std::max(ic, oc) >= 256);
        int sp_min_blk = 24;
        int sp_max_blk = (are_small_large_channels ? 100 : 50);

        auto get_score = [&](int oh_blk, int ow_blk) {
            int sp_blk = oh_blk * ow_blk;
            int oh_padded = utils::rnd_up(oh, oh_blk);
            int ow_padded = utils::rnd_up(ow, ow_blk);

            double extra_work
                    = (oh_padded * ow_padded - oh * ow) / double(oh * ow);
            // ohw_eff == 0: no useful computation
            // ohw_eff == 1: all computation is useful
            double ohw_eff = 1 - std::min(extra_work, 1.0);
            int score = int(ohw_eff * 10000);

            // Prefer [sp_min_blk; sp_max_blk] range for the total spatial size.
            bool sp_size_ok = (sp_blk >= sp_min_blk && sp_blk <= sp_max_blk);

            if (hw >= ngen::HW::XeHPC) {
                bool sp_block_ok = false;
                // Avoid OH blocking when OW blocking is enabled and big enough (to
                // avoid code explosion due after mandatory unrolling of inner
                // iterations). Exception: when OH/OW are fully blocked - even with
                // code explosion such blocks may give the best performance.
                sp_block_ok |= (oh_blk == 1 || ow_blk <= 2);
                sp_block_ok |= (oh_blk == oh && ow_blk == ow);
                if (sp_size_ok && sp_block_ok) {
                    double sp_range = sp_max_blk - sp_min_blk;
                    double sp_score = (sp_blk - sp_min_blk) / sp_range * 100;
                    score += sp_score;
                }
            } else if (sp_size_ok) {
                score += 100;
            }
            return score;
        };

        int max_score = 0;
        for (int oh_blk = 1; oh_blk <= sp_max_blk; oh_blk++) {
            for (int ow_blk = ow_thr_blk; ow_blk <= sp_max_blk;
                    ow_blk += ow_thr_blk) {
                int score = get_score(oh_blk, ow_blk);
                if (score > max_score) {
                    oh_tg_blk = oh_blk;
                    ow_tg_blk = ow_blk;
                    max_score = score;
                }
            }
        }

#ifdef GEN_CONV_DEBUG
        od_tg_blk = getenv_int("od_tg_blk", od_tg_blk);
        oh_tg_blk = getenv_int("oh_tg_blk", oh_tg_blk);
        ow_tg_blk = getenv_int("ow_tg_blk", ow_tg_blk);
#endif
    }

    status_t init_common_config() {
        using namespace ir_utils;

        use_preload = true;

        if (hw <= ngen::HW::XeLP) use_preload = false;

        // No SLM buffering by default (will be enabled later).
        disable_slm_buffering();

        // No prefetch by default (will be enabled later).
        disable_prefetch();

        do_b_reduction = false;
        pad_slm = true;
        assign_sbids = is_dpas_fma();
        do_loop_unroll = hw > ngen::HW::XeLP;
        reduce_grf_usage = true;
        do_atomic_update = false;
        reuse_headers = hw <= ngen::HW::XeLP;
        optimize_strided = false;
        a_sub_tiles = 1;
        b_sub_tiles = 1;

#ifdef GEN_CONV_DEBUG
        use_preload = getenv_bool("use_preload", use_preload);
        pad_slm = getenv_bool("pad_slm", pad_slm);
        assign_sbids = getenv_bool("assign_sbids", assign_sbids);
        do_loop_unroll = getenv_bool("do_loop_unroll", do_loop_unroll);
        reduce_grf_usage = getenv_bool("reduce_grf_usage", reduce_grf_usage);
        allow_grf_reorder = getenv_bool("allow_grf_reorder", allow_grf_reorder);
        reuse_headers = getenv_bool("reuse_headers", reuse_headers);
        a_sub_tiles = getenv_int("a_sub_tiles", a_sub_tiles);
        b_sub_tiles = getenv_int("b_sub_tiles", b_sub_tiles);
#endif

        return status::success;
    }

    bool post_ops_ok(const convolution_pd_t *pd) const {
        auto *attr = pd->attr();

        if (is_fwd || is_bwd_d) {
            auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops
                    | primitive_attr_t::skip_mask_t::oscale_runtime
                    | primitive_attr_t::skip_mask_t::sum_dt;
            if (!attr->has_default_values(attr_skip_mask)) return false;
        } else {
            if (!attr->has_default_values()) return false;
        }

        if (!attr->output_scales_.has_default_values()) {
            // Only common and per_oc output scales were tested.
            if (!utils::one_of(attr->output_scales_.mask_, 0, (1 << 1)))
                return false;
        }
        for (int i = 0; i < attr->post_ops_.len(); i++) {
            auto &po = attr->post_ops_.entry_[i];
            if (po.is_eltwise()) {
                if (!jit_eltwise_injector_f32_is_supported(po.eltwise.alg))
                    return false;
            } else if (po.is_binary()) {
                int mask = utils::get_dims_mask(pd->invariant_dst_md()->dims,
                        po.binary.src1_desc.dims, ndims);
                // per_oc broadcast is always supported.
                if ((mask & (1 << 1)) == 0) continue;
                auto rhs_layout = layout_t(po.binary.src1_desc);
                // No blocks means it's a scalar, can be always loaded.
                if (rhs_layout.blocks().empty()) return true;

                auto rhs0 = rhs_layout.blocks()[0];
                int block_bytes = rhs0.block * rhs_layout.type().size();
                // Innermost block must:
                // - be across output channels
                // - be dense
                // - aligned to 32 bytes (for HWord loads)
                if (rhs0.dim_idx != 1 || dim_t(rhs0.stride) != 1
                        || block_bytes % 32 != 0)
                    return false;
            }
        }
        return true;
    }

    bool hw_ok(const engine_t *engine) const {
        auto *compute_engine
                = utils::downcast<const compute::compute_engine_t *>(engine);
        if (regs == 256 && !compute_engine->mayiuse_large_grf_mode())
            return false;
        return true;
    }

    bool data_types_ok() const {
        bool is_bf16 = utils::one_of(data_type::bf16, src_data_type,
                wei_data_type, dst_data_type, bia_data_type);
        if (is_bf16 && hw <= ngen::HW::XeLP) return false;

        if (is_fwd) return true;
        if (is_bwd_d) return true;
        if (is_bwd_w) {
            bool ok = true;
            ok &= (src_data_type == data_type::bf16
                    || src_data_type == data_type::f32);
            ok &= (dst_data_type == src_data_type);
            ok &= utils::one_of(wei_data_type, src_data_type, data_type::f32);

            if (with_bias) {
                ok &= utils::one_of(
                        bia_data_type, src_data_type, data_type::f32);
            }
            return ok;
        }
        return false;
    }

    bool is_s32_accumulator() const { return acc_data_type == data_type::s32; }
    bool is_f32_conv() const {
        return utils::everyone_is(src_data_type, wei_data_type, data_type::f32);
    }
    bool is_int8_dst() const {
        return utils::one_of(dst_data_type, data_type::s8, data_type::u8);
    }
    bool is_small_ic() const { return ic < simd_size; }
    bool is_dpas_fma() const {
        return utils::one_of(fma_kind, fma_kind_t::dpas, fma_kind_t::dpasw);
    }

    int grf_size() const { return ngen::GRF::bytes(hw); }

    compute::nd_range_t nd_range() const {
        size_t gws[3];
        size_t lws[3];
        for (int i = 0; i < 3; i++) {
            lws[i] = tg_grid_dim[i] * (i == 0 ? simd_size : 1);
            gws[i] = kernel_grid_dim[i] * lws[i];
        }
        return compute::nd_range_t(gws, lws);
    }

    std::string str() const {
        using namespace ir_utils;

        std::ostringstream oss;
        // clang-format off
        oss << "  Problem:                    " << desc_str() << std::endl;
        oss << "  Source layout:              " << tensor_config.compute_layout("src") << std::endl;
        oss << "  Weights layout:             " << tensor_config.compute_layout("wei") << std::endl;
        oss << "  Destination layout:         " << tensor_config.compute_layout("dst") << std::endl;
        oss << "  MB TG block:                " << mb_tg_blk << std::endl;
        oss << "  OD TG block:                " << od_tg_blk << std::endl;
        oss << "  OH TG block:                " << oh_tg_blk << std::endl;
        oss << "  OW TG block:                " << ow_tg_blk << std::endl;
        oss << "  OC TG block:                " << oc_tg_blk << std::endl;
        oss << "  Kernel grid:                " << make_seq_print_helper(kernel_grid_dim, " x ") << std::endl;
        oss << "  Thread group:               " << make_seq_print_helper(tg_grid_dim, " x ") << std::endl;
        oss << "  FMA kind:                   " << fma_kind::to_string(fma_kind) << std::endl;
        oss << "  Use SLM for A:              " << to_string(use_a_slm) << std::endl;
        oss << "  Use SLM for B:              " << to_string(use_b_slm) << std::endl;
        oss << "  SLM buffers:                " << slm_bufs << std::endl;
        oss << "  GMEM to SLM, GRF buffers:   " << gmem_bufs << std::endl;
        oss << "  Pad SLM:                    " << to_string(pad_slm) << std::endl;
        oss << "  Use prefetch:               " << to_string(use_prefetch) << std::endl;
        oss << "  Prefetch buffers:           " << prefetch_bufs << std::endl;
        oss << "  Do loop unroll:             " << to_string(do_loop_unroll) << std::endl;
        oss << "  Assign SBIDs:               " << to_string(assign_sbids) << std::endl;
        oss << "  Reduce GRF usage:           " << to_string(reduce_grf_usage) << std::endl;
        oss << "  Reuse headers:              " << to_string(reuse_headers) << std::endl;
        oss << "  Allow GRF reorder:          " << to_string(allow_grf_reorder) << std::endl;
        oss << "  A sub-tiles:                " << a_sub_tiles << std::endl;
        oss << "  B sub-tiles:                " << b_sub_tiles << std::endl;
        // clang-format on
        return oss.str();
    }

    data_type_t a_data_type;
    data_type_t b_data_type;
    data_type_t c_data_type;
    data_type_t acc_data_type;

    ngen::HW hw = ngen::HW::Unknown;
    int simd_size; // SIMD width.
    int regs; // Number of registers.

    // Thread group dimensions (thread group grid).
    std::array<int, 3> tg_grid_dim;

    // Number of thread groups across dimensions (kernel grid).
    std::array<int, 3> kernel_grid_dim;

    // Number of thread group blocks across problem dimensions.
    int g_tg_dim;
    int ic_tg_dim;
    int iw_tg_dim;
    int kw_tg_dim;
    int mb_tg_dim;
    int oc_tg_dim;
    int od_tg_dim;
    int oh_tg_dim;
    int ow_tg_dim;

    // Block sizes per thread group.
    int g_tg_blk;
    int ic_tg_blk;
    int iw_tg_blk;
    int kw_tg_blk;
    int mb_tg_blk;
    int oc_tg_blk;
    int od_tg_blk;
    int oh_tg_blk;
    int ow_tg_blk;

    // Number of thread blocks across problem dimensions.
    int ic_thr_dim;
    int iw_thr_dim;
    int mb_thr_dim;
    int oc_thr_dim;
    int ow_thr_dim;

    // Block sizes per thread.
    int g_thr_blk;
    int ic_thr_blk;
    int iw_thr_blk;
    int mb_thr_blk;
    int oc_thr_blk;
    int ow_thr_blk;

    // Block sizes per iteration.
    int ic_blk;
    int kw_blk;
    int mb_blk;
    int oc_blk;

    // Block sizes in GEMM notation.
    int b_blk;
    int m_tg_blk;
    int n_tg_blk;
    int k_blk;

    // Unroll sizes.
    int mb_unroll;
    int ow_unroll;

    bool do_b_reduction;

    fma_kind_t fma_kind; // Which instruction backend to use.

    bool use_preload; // Whether to use SLM or prefetch.
    bool use_a_slm; // Whether to use SLM for A.
    bool use_b_slm; // Whether to use SLM for B.
    bool use_prefetch; // Whether to use prefetch for A and B.
    bool pad_slm; // Whether to pad SLM to avoid write conflicts.
    bool assign_sbids; // Whether to manually assign SBID tokens.
    int slm_bufs; // Number of SLM buffers to use.
    int gmem_bufs; // Number of GRF buffers to use for GMEM -> SLM copy.
    int prefetch_bufs; // Number of prefetch buffers for A and B.
    bool do_loop_unroll; // Whether to fully unroll inner loops.
    bool reduce_grf_usage; // Whether to try to reduce GRF usage based on heuristics.
    bool allow_grf_reorder; // Whether to allow GRF reorders to FMA-friendly layouts.
    bool do_atomic_update; // Whether to use atomics during C update.
    bool reuse_headers; // Whether to reuse header messages to reduce GRF usage.
    bool optimize_strided; // Apply special optimization for strided BWD_D convolution.

    // Sub-tiles to split into for the inner A x B multiplication:
    // for i in range(0, a_sub_tiles):
    //     A_i = load(...)
    //     for j in range(0, b_sub_tiles):
    //         B_j = load(...)
    //         C_i_j += A_i * B_j
    //
    // GRF buffers for A_i and B_j are reused. Factors greater than one help to
    // reduce GRF usage.
    int a_sub_tiles;
    int b_sub_tiles;

private:
    int init_fwd_ic_thr_dim(engine_t *engine, int mb_thr_blk, int oc_thr_blk,
            int ow_thr_blk, int ic_blk) const {
        if (mb_thr_blk > 1) return 1;

        int ic_blocks = utils::div_up(ic, ic_blk);
        int reduction_blocks = ic_blocks * kd * kh * kw;

        int oc_nthr = utils::div_up(oc, oc_thr_blk);
        int ow_nthr = utils::div_up(ow, ow_thr_blk);
        int mb_nthr = utils::div_up(mb, mb_thr_blk);
        int nthr = mb_nthr * oc_nthr * od * oh * ow_nthr;

        auto *compute_engine
                = utils::downcast<const compute::compute_engine_t *>(engine);
        int eus = compute_engine->device_info()->eu_count();

        int ret_ic_thr_dim = 1;
        if (!is_small_ic() && reduction_blocks >= 16 && (nthr < eus)) {
            ret_ic_thr_dim = ir_utils::max_divisor(ic_blocks, {1, 2, 4, 8});

            // If reduction is too small, limit k-slicing.
            int reduction_threshold = 32;
            if (reduction_blocks < reduction_threshold) {
                int max_ic_thr_dim = utils::div_up(eus, nthr);
                max_ic_thr_dim = (1 << math::ilog2q(max_ic_thr_dim));
                ret_ic_thr_dim = std::min(ret_ic_thr_dim, max_ic_thr_dim);
            }
        }
        return ret_ic_thr_dim;
    }

    status_t init_hw(engine_t *engine) {
        using namespace compute;

        auto compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);
        auto device_info = compute_engine->device_info();

        switch (device_info->gpu_arch()) {
            case gpu_arch_t::gen9: hw = ngen::HW::Gen9; break;
            case gpu_arch_t::xe_lp: hw = ngen::HW::XeLP; break;
            case gpu_arch_t::xe_hp: hw = ngen::HW::XeHP; break;
            case gpu_arch_t::xe_hpg: hw = ngen::HW::XeHPG; break;
            case gpu_arch_t::xe_hpc: hw = ngen::HW::XeHPC; break;
            default: return status::unimplemented;
        }
        return status::success;
    }

    // Initializes A/B/C data types (GEMM notation: C += A * B) according to
    // the following convention:
    // FWD:        src -> A,      wei -> B,      dst -> C
    // BWD_D: diff_dst -> A,      wei -> B, diff_src -> C
    // BWD_W:      src -> A, diff_dst -> B, diff_wei -> C
    status_t init_abc_data_types() {
        if (is_fwd) {
            a_data_type = src_data_type;
            b_data_type = wei_data_type;
            c_data_type = dst_data_type;
        } else if (is_bwd_d) {
            a_data_type = dst_data_type;
            b_data_type = wei_data_type;
            c_data_type = src_data_type;
        } else if (is_bwd_w) {
            a_data_type = src_data_type;
            b_data_type = dst_data_type;
            // Always use f32 for accumulation/storing in the main kernel.
            c_data_type = data_type::f32;
        } else {
            ir_error_not_expected();
        }
        return status::success;
    }

    status_t init_acc_data_type() {
        auto a = a_data_type;
        auto b = b_data_type;
        auto c = c_data_type;
        if (utils::one_of(a, data_type::s8, data_type::u8)
                && utils::one_of(b, data_type::s8, data_type::u8)) {
            acc_data_type = data_type::s32;
            return status::success;
        }
        if (utils::everyone_is(data_type::f16, a, b)
                || utils::everyone_is(data_type::bf16, a, b)) {
            acc_data_type = data_type::f32;
            return status::success;
        }
        if (utils::everyone_is(data_type::f32, a, b, c)) {
            acc_data_type = data_type::f32;
            return status::success;
        }
        return status::unimplemented;
    }

    status_t init_fma_kind() {
        fma_kind = fma_kind::get_supported_kind(
                hw, a_data_type, b_data_type, acc_data_type);

        simd_size = fma_kind::get_simd_size(
                hw, fma_kind, a_data_type, b_data_type, acc_data_type);

        bool use_mad = false;
        if (is_small_ic() && !is_dw) {
            if (is_fwd && (kw != 7 || mb % 8 != 0))
                use_mad = true;
            else if (is_bwd_d)
                use_mad = true;
        } else if (is_dw) {
            use_mad = true;
        }

        if (use_mad) {
            fma_kind = fma_kind_t::mad;
            simd_size = fma_kind::get_simd_size(
                    hw, fma_kind, a_data_type, b_data_type, acc_data_type);
        }

#ifdef GEN_CONV_DEBUG
        fma_kind = fma_kind::from_string(ir_utils::getenv_str(
                "fma_kind", fma_kind::to_string(fma_kind)));
        simd_size = fma_kind::get_simd_size(
                hw, fma_kind, a_data_type, b_data_type, acc_data_type);

#endif
        if (fma_kind == fma_kind_t::unknown) return status::unimplemented;

        // Disable using mad instruction backend until performance parity is
        // reached with OpenCL kernels.
        if (fma_kind == fma_kind_t::mad) {
            if (hw < ngen::HW::XeHP) return status::unimplemented;
            if (is_bwd_d) {
                if (!is_f32_conv()) return status::unimplemented;
                if (is_small_ic()) return status::unimplemented;
                return status::success;
            }
        }

        return status::success;
    }

    status_t init_data_layouts(convolution_pd_t *conv_pd) {
        // Compute layout tags and user layout tags. If a compute layout is
        // different from a user layout then an extra pre/post reorder will be
        // executed before/after convolution.
        std::string src_tag, user_src_tag;
        std::string wei_tag, user_wei_tag;
        std::string dst_tag, user_dst_tag;

        const bool is_wei16aXb = hw >= ngen::HW::XeHPC;
        assert(hw != ngen::HW::Unknown);
        bool is_mb_block = mb >= 16;

        // Src/Dst buffers should generally be the same format to avoid reorders
        // between FWD, BWD_D, and BWD_W.
        if (is_small_ic() && !is_dw) {
            src_tag = is_s32_accumulator() ? "ABx8a4b" : "ABx8a2b";
        } else if (fma_kind == fma_kind_t::mad) {
            if (is_s32_accumulator()) {
                src_tag = (!is_mb_block ? "aBx32b" : "ABx32a32b");
            } else {
                src_tag = (!is_mb_block ? "aBx16b" : "ABx32a16b");
            }
            if (is_fwd) {
                int max_simd_size = 16;
                if (simd_size > max_simd_size) simd_size = max_simd_size;
            }
        } else if (is_s32_accumulator()) {
            src_tag = (!is_mb_block ? "aBx32b" : "ABx32a32b");
        } else {
            src_tag = (!is_mb_block ? "aBx16b" : "ABx32a16b");
        }

        if (fma_kind == fma_kind_t::mad) {
            if (is_dw) {
                if (is_int8_dst()) {
                    dst_tag = (!is_mb_block ? "aBx32b" : "ABx32a32b");
                } else {
                    dst_tag = (!is_mb_block ? "aBx16b" : "ABx32a16b");
                }
            } else {
                dst_tag = (!is_mb_block ? "aBx16b" : "ABx32a16b");
            }
            if (is_bwd_d) {
                int max_simd_size = 16;
                if (simd_size > max_simd_size) simd_size = max_simd_size;
            }
        } else if (is_int8_dst()) {
            dst_tag = (!is_mb_block ? "aBx32b" : "ABx32a32b");
        } else {
            dst_tag = (!is_mb_block ? "aBx16b" : "ABx32a16b");
        }

        // Weight reorders are generally small, so reordering weights between
        // FWD and BWD_D/BWD_W implementations for optimization purposes makes
        // sense.
        if (is_fwd) {
            if (is_small_ic() && !is_dw) {
                if (fma_kind == fma_kind_t::mad)
                    wei_tag = "bAx16a";
                else if (is_s32_accumulator())
                    wei_tag = is_wei16aXb ? "ABx16a4b" : "ABx8a4b";
                else
                    wei_tag = is_wei16aXb ? "ABx16a2b" : "ABx8a2b";
            } else {
                if (is_dw) {
                    if (is_s32_accumulator())
                        wei_tag = "Abcx32a";
                    else
                        wei_tag = "Abcx16a";
                } else if (fma_kind == fma_kind_t::mad) {
                    wei_tag = "BAx16b16a";
                } else if (is_s32_accumulator()) {
                    wei_tag = is_wei16aXb ? "ABx2a8b16a4b" : "ABx4a8b8a4b";
                } else {
                    wei_tag = is_wei16aXb ? "ABx8b16a2b" : "ABx2a8b8a2b";
                }
            }
        } else if (is_bwd_d) {
            // Set user_wei_tag to match forward for dpas to be able to reuse
            // the same weights buffer/layout on the user side. Compute layout
            // is different to match dpas so an extra reorder will be used.
            if (fma_kind == fma_kind_t::mad)
                wei_tag = "ABx16a16b";
            else if (is_s32_accumulator()) {
                user_wei_tag = is_wei16aXb ? "ABx2a8b16a4b" : "ABx4a8b8a4b";
                wei_tag = is_wei16aXb ? "ABx2b8a16b4a" : "ABx4b8a8b4a";
            } else {
                user_wei_tag = is_wei16aXb ? "ABx8b16a2b" : "ABx2a8b8a2b";
                wei_tag = is_wei16aXb ? "ABx8a16b2a" : "ABx2b8a8b2a";
            }
        } else if (is_bwd_w) {
            if (is_small_ic()) {
                wei_tag = is_wei16aXb ? "ABx16a2b" : "ABx8a2b";
            } else {
                wei_tag = is_wei16aXb ? "ABx8b16a2b" : "ABx2a8b8a2b";
            }
        }

        if (user_src_tag.empty()) user_src_tag = src_tag;
        if (user_wei_tag.empty()) user_wei_tag = wei_tag;
        if (user_dst_tag.empty()) user_dst_tag = dst_tag;

        if (with_groups && !is_dw) {
            wei_tag = prepend_groups_to_tag(wei_tag);
            user_wei_tag = prepend_groups_to_tag(user_wei_tag);
        }

#ifdef GEN_CONV_DEBUG
        src_tag = ir_utils::getenv_str("stag", src_tag);
        wei_tag = ir_utils::getenv_str("wtag", wei_tag);
        dst_tag = ir_utils::getenv_str("dtag", dst_tag);

        user_src_tag = ir_utils::getenv_str("user_stag", user_src_tag);
        user_wei_tag = ir_utils::getenv_str("user_wtag", user_wei_tag);
        user_dst_tag = ir_utils::getenv_str("user_dtag", user_dst_tag);
#endif

        auto &src_md = *conv_pd->invariant_src_md();
        auto &wei_md = *conv_pd->invariant_wei_md();
        auto &dst_md = *conv_pd->invariant_dst_md();
        auto &bia_md = *conv_pd->invariant_bia_md();

        // Select user layouts.
        auto src_layout = init_layout(src_md, user_src_tag);
        auto wei_layout = init_layout(wei_md, user_wei_tag);
        auto dst_layout = init_layout(dst_md, user_dst_tag);

        layout_t bia_layout;
        if (with_bias) bia_layout = init_layout(bia_md, "a");

        // Validate layouts.
        bool is_src_nhwc = is_nhwc("src", conv_pd, src_layout);
        bool is_dst_nhwc = is_nhwc("dst", conv_pd, dst_layout);
        if (is_src_nhwc != is_dst_nhwc) return status::unimplemented;

        if (is_src_nhwc) {
            if (is_bwd_w) return status::unimplemented;

            src_tag = user_src_tag = "axb";
            dst_tag = user_dst_tag = "axb";

            // HWord loads require 32 byte alignment. For NHWC layout it means
            // input/output channels must be multiples of 32 bytes.
            size_t ic_bytes = ic * types::data_type_size(src_data_type);
            size_t oc_bytes = oc * types::data_type_size(dst_data_type);
            if (ic_bytes % 32 != 0 || oc_bytes % 32 != 0)
                return status::unimplemented;
        }
        if (!src_layout.is_strictly_equal(make_layout(src_md, user_src_tag)))
            return status::unimplemented;
        if (!dst_layout.is_strictly_equal(make_layout(dst_md, user_dst_tag)))
            return status::unimplemented;
        if (!wei_layout.is_strictly_equal(make_layout(wei_md, user_wei_tag)))
            return status::unimplemented;

        tensor_config.add_tensor("src", src_arg_key(), is_src_input(),
                is_src_output(), src_layout);
        tensor_config.add_tensor("wei", wei_arg_key(), is_wei_input(),
                is_wei_output(), wei_layout);
        if (with_bias)
            tensor_config.add_tensor("bia", bia_arg_key(), is_bia_input(),
                    is_bia_output(), bia_layout);
        tensor_config.add_tensor("dst", dst_arg_key(), is_dst_input(),
                is_dst_output(), dst_layout);

        if (src_tag != user_src_tag)
            tensor_config.set_compute_layout("src", layout_t(src_md, src_tag));

        if (wei_tag != user_wei_tag)
            tensor_config.set_compute_layout("wei", layout_t(wei_md, wei_tag));

        if (dst_tag != user_dst_tag)
            tensor_config.set_compute_layout("dst", layout_t(dst_md, dst_tag));

        return status::success;
    }

    status_t init_extra_tensor_layouts(const convolution_pd_t *conv_pd) {
        auto *attr = conv_pd->attr();
        bool with_oscales = !attr->output_scales_.has_default_values();
        if (with_oscales) {
            std::vector<dim_t> dims = {attr->output_scales_.count_};
            layout_t oscales_layout(type_t::f32(), 0, dims);
            int arg_key = -1;
            if (!attr->output_scales_.defined())
                arg_key = DNNL_ARG_ATTR_OUTPUT_SCALES;
            tensor_config.add_tensor("oscales", arg_key, /*is_input=*/true,
                    /*is_output=*/false, oscales_layout);
        }

        for (int i = 0; i < attr->post_ops_.len(); i++) {
            auto &po = attr->post_ops_.entry_[i];
            if (po.is_eltwise() || po.is_sum(/*require_scale_one=*/false)) {
                // No extra tensors.
            } else if (po.is_binary()) {
                auto layout = make_layout(po.binary.src1_desc);
                int arg_key
                        = DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1;
                tensor_config.add_tensor("binary_rhs_" + std::to_string(i),
                        arg_key, /*is_input=*/true,
                        /*is_output=*/false, layout);
            } else {
                ir_error_not_expected();
            }
        }
        return status::success;
    }

    bool can_optimize_strided_bwd_d() const {
        if (iw_thr_blk > 1) return false;
        if (is_stride1()) return false;
        if (iw % sw != 0) return false;
        return true;
    }

    void enable_slm_buffering() {
        using namespace ir_utils;

        use_a_slm = (tg_grid_dim[0] > 1);
        use_b_slm = (tg_grid_dim[1] > 1);
        if (use_a_slm || use_b_slm) {
            int pref_slm_bufs = (tg_grid_dim[0] * tg_grid_dim[1] <= 8 ? 2 : 3);
            if (do_loop_unroll) {
                slm_bufs = pref_slm_bufs;
                gmem_bufs = (is_dpas_fma() ? 2 : 1);
            } else {
                // Double/triple SLM buffering is not supported when only one
                // matrix is SLM-buffered.
                slm_bufs = (use_a_slm == use_b_slm ? pref_slm_bufs : 1);
                gmem_bufs = 1;
            }
        } else {
            slm_bufs = 0;
            gmem_bufs = 0;
        }
#ifdef GEN_CONV_DEBUG
        use_a_slm = getenv_bool("use_a_slm", use_a_slm);
        use_b_slm = getenv_bool("use_b_slm", use_b_slm);
        slm_bufs = getenv_int("slm_bufs", slm_bufs);
        gmem_bufs = getenv_int("gmem_bufs", gmem_bufs);
#endif
    }

    void enable_prefetch() {
        using namespace ir_utils;

        use_prefetch = true;
        prefetch_bufs = is_bwd_w ? 2 : 3;
#ifdef GEN_CONV_DEBUG
        use_prefetch = getenv_bool("use_prefetch", use_prefetch);
        prefetch_bufs = getenv_int("prefetch_bufs", prefetch_bufs);
#endif
    }

    void disable_slm_buffering() {
        use_a_slm = false;
        use_b_slm = false;
        slm_bufs = 0;
        gmem_bufs = 0;
    }

    void disable_prefetch() {
        use_prefetch = false;
        prefetch_bufs = 0;
    }

    // Overwrites parameters that are implied by other parameters.
    void fixup_inference_consistency() {
        // Can't reuse headers with loop unroll and post-increment offset updates.
        if (reuse_headers) do_loop_unroll = false;

        bool prefer_prefetch = false;
        if (hw >= ngen::HW::XeHPC) prefer_prefetch = true;

        if (use_preload) {
            // Prefetches are only supported with loop unrolling.
            if (prefer_prefetch && do_loop_unroll) {
                enable_prefetch();
            } else {
                enable_slm_buffering();
            }
        }
        // Downgrade dpasw -> dpas for some cases.
        if (fma_kind == fma_kind_t::dpasw) {
            // dpasw is executed by fused EUs (across X thread group
            // dimension). Do not use dpasw if X is uneven.
            if (tg_grid_dim[0] % 2 != 0) fma_kind = fma_kind_t::dpas;
            // dpasw can't be generated in case of direct load from GMEM and reorder.
            if (is_bwd_w && allow_grf_reorder && (!use_a_slm || !use_b_slm))
                fma_kind = fma_kind_t::dpas;
        }
    }

    bool try_reduce_grf_usage() {
        if (!reduce_grf_usage) return true;

        // TODO: improve estimate register count, it fails to account for tmp
        // values like mask_registers among other things.
        double reg_factor = is_bwd_w ? 0.875 : 0.95;
        int max_regs = int(regs * reg_factor);
        int regs = estimate_register_count();
        if (regs <= max_regs) return true;

        // Try to disable GRF buffering.
        if (gmem_bufs > 1) {
            gmem_bufs = 1;
            int regs = estimate_register_count();
            if (regs <= max_regs) return true;
        }

        // Try to use sub-tiles for B.
        int n_thr_blk = utils::div_up(n_tg_blk, tg_grid_dim[0]);
        int max_b_sub_tiles
                = std::min((use_b_slm ? 4 : 2), n_thr_blk / simd_size);
        // XXX: avoid layout mismatch for B loads
        if (hw >= ngen::HW::XeHPC && is_bwd_w) max_b_sub_tiles = 2;
        while (b_sub_tiles < max_b_sub_tiles) {
            b_sub_tiles *= 2;
            int regs = estimate_register_count();
            if (regs <= max_regs) return true;
        }

        // Try to use double SLM buffering.
        if (slm_bufs == 3) {
            slm_bufs = 2;
            int regs = estimate_register_count();
            if (regs <= max_regs) return true;
        }

        // Try to use single SLM buffering.
        if (slm_bufs == 2) {
            slm_bufs = 1;
            int regs = estimate_register_count();
            if (regs <= max_regs) return true;
        }

        // Last resort settings to reduce GRF usage.
        reuse_headers = true;
        do_loop_unroll = false;

        return estimate_register_count() <= max_regs;
    }

    int estimate_register_count() const {
        int reg_bytes = ngen::GRF::bytes(hw);

        // Assume 8 HWord per GMEM load for double-blocked layouts and 1 HWord
        // otherwise.
        int hword_bytes = 32;
        int a_gmem_msg_bytes
                = (a_layout().is_n_blocked(2) ? 8 : 1) * hword_bytes;
        int b_gmem_msg_bytes
                = (b_layout().is_n_blocked(2) ? 8 : 1) * hword_bytes;

        // Assume 8 HWords per SLM load/store.
        int slm_msg_bytes = 256;

        int nthr = tg_grid_dim[0] * tg_grid_dim[1];
        int m_thr_blk = utils::div_up(m_tg_blk, tg_grid_dim[1]);
        int n_thr_blk = utils::div_up(n_tg_blk, tg_grid_dim[0]);
        int k_thr_blk = k_blk;

        int a_size = int(types::data_type_size(a_data_type));
        int b_size = int(types::data_type_size(b_data_type));
        int acc_size = int(types::data_type_size(acc_data_type));

        // Registers for C += A * B operation.
        int a_tile_bytes = m_thr_blk * k_thr_blk * a_size;
        int b_tile_bytes = k_thr_blk * n_thr_blk * b_size;
        int a_bytes = utils::div_up(a_tile_bytes, a_sub_tiles);
        int b_bytes = utils::div_up(b_tile_bytes, b_sub_tiles);
        int acc_bytes = m_thr_blk * n_thr_blk * acc_size;

        int a_regs = utils::div_up(a_bytes, reg_bytes);
        int b_regs = utils::div_up(b_bytes, reg_bytes);
        int acc_regs = utils::div_up(acc_bytes, reg_bytes);

        int a_headers = utils::div_up(
                a_tile_bytes, use_a_slm ? slm_msg_bytes : a_gmem_msg_bytes);
        int b_headers = utils::div_up(
                b_tile_bytes, use_b_slm ? slm_msg_bytes : b_gmem_msg_bytes);

        if (fma_kind == fma_kind_t::dpasw) {
            // dpasw reuses registers between fused threads across tg0. M is
            // split across tg1, N is split across tg0 so dpasw allows to share
            // matrix A which is is (M x K).
            a_regs = utils::div_up(a_regs, 2);
            a_headers = utils::div_up(a_headers, 2);
        }

        // Size of A/B thread blocks when split full A/B TG blocks across all
        // threads in TG.
        int a_tg_per_thr_bytes = utils::div_up(m_tg_blk * k_blk * a_size, nthr);
        int b_tg_per_thr_bytes = utils::div_up(k_blk * n_tg_blk * b_size, nthr);

        // Temporary registers for GMEM -> SLM load.
        int a_g2s_bytes = (use_a_slm ? a_tg_per_thr_bytes : 0);
        int b_g2s_bytes = (use_b_slm ? b_tg_per_thr_bytes : 0);

        // Account for dedicated headers for prefetches.
        if (use_prefetch) {
            a_headers += utils::div_up(a_tg_per_thr_bytes, a_gmem_msg_bytes);
            b_headers += utils::div_up(b_tg_per_thr_bytes, b_gmem_msg_bytes);
        }

        int a_g2s_regs = utils::div_up(a_g2s_bytes, reg_bytes);
        int b_g2s_regs = utils::div_up(b_g2s_bytes, reg_bytes);

        // Two sets of headers for GMEM -> GRF and GRF -> SLM.
        int a_g2s_headers = utils::div_up(a_g2s_bytes, a_gmem_msg_bytes)
                + utils::div_up(a_g2s_bytes, slm_msg_bytes);
        int b_g2s_headers = utils::div_up(b_g2s_bytes, b_gmem_msg_bytes)
                + utils::div_up(b_g2s_bytes, slm_msg_bytes);

        // Extra registers for GRF <-> GRF reorders.
        int reorder_regs = 0;

        // Assume A/B need reorders to temporary buffers.
        if (allow_grf_reorder) {
            if (is_bwd_w) {
                // Hardcode for now, this is the upper bound for the temporary
                // buffer size for BWD_W.
                int bwd_w_reorder_regs = 16;
                reorder_regs += bwd_w_reorder_regs;
            }

            int ab_reorder_regs = 0;

            if (use_a_slm) {
                ab_reorder_regs = std::max(ab_reorder_regs, a_g2s_regs);
            } else {
                int a_reorder_regs = a_regs;
                // Loads must be aligned to a GRF boundary, account for cases
                // when the load size is less than the register size.
                if (a_gmem_msg_bytes < reg_bytes) {
                    a_reorder_regs
                            *= utils::div_up(reg_bytes, a_gmem_msg_bytes);
                }
                ab_reorder_regs = std::max(ab_reorder_regs, a_reorder_regs);
            }
            if (use_b_slm) {
                ab_reorder_regs = std::max(ab_reorder_regs, b_g2s_regs);
            } else {
                int b_reorder_regs = b_regs;
                // Loads must be aligned to a GRF boundary, account for cases
                // when the load size is less than the register size.
                if (b_gmem_msg_bytes < reg_bytes) {
                    b_reorder_regs
                            *= utils::div_up(reg_bytes, b_gmem_msg_bytes);
                }
                ab_reorder_regs = std::max(ab_reorder_regs, b_reorder_regs);
            }
            reorder_regs += ab_reorder_regs;
        }

        int g2s_regs = gmem_bufs * (a_g2s_regs + b_g2s_regs);
        int g2s_headers = a_g2s_headers + b_g2s_headers;

        int data_regs = a_regs + b_regs + acc_regs + g2s_regs;
        int header_regs = a_headers + b_headers + g2s_headers;
        if (reuse_headers) header_regs = 1;

        int estimated_regs = data_regs + reorder_regs + header_regs;

        return estimated_regs;
    }

    const layout_t &a_layout() const {
        if (is_fwd) return tensor_config.compute_layout("src");
        if (is_bwd_d) return tensor_config.compute_layout("dst");
        return tensor_config.compute_layout("src");
    }

    const layout_t &b_layout() const {
        if (is_fwd) return tensor_config.compute_layout("wei");
        if (is_bwd_d) return tensor_config.compute_layout("wei");
        return tensor_config.compute_layout("dst");
    }

    int src_arg_key() const {
        if (is_fwd) return DNNL_ARG_SRC;
        if (is_bwd_d) return DNNL_ARG_DIFF_SRC;
        if (is_bwd_w) return DNNL_ARG_SRC;
        ir_error_not_expected();
        return -1;
    }

    bool is_src_input() const { return is_fwd || is_bwd_w; }
    bool is_src_output() const { return is_bwd_d; }

    int wei_arg_key() const {
        if (is_fwd) return DNNL_ARG_WEIGHTS;
        if (is_bwd_d) return DNNL_ARG_WEIGHTS;
        if (is_bwd_w) return DNNL_ARG_DIFF_WEIGHTS;
        ir_error_not_expected();
        return -1;
    }

    bool is_wei_input() const { return is_fwd || is_bwd_d; }
    bool is_wei_output() const { return is_bwd_w; }

    int bia_arg_key() const {
        if (is_fwd) return DNNL_ARG_BIAS;
        if (is_bwd_d) return DNNL_ARG_BIAS;
        if (is_bwd_w) return DNNL_ARG_DIFF_BIAS;
        ir_error_not_expected();
        return -1;
    }

    bool is_bia_input() const { return is_fwd || is_bwd_d; }
    bool is_bia_output() const { return is_bwd_w; }

    int dst_arg_key() const {
        if (is_fwd) return DNNL_ARG_DST;
        if (is_bwd_d) return DNNL_ARG_DIFF_DST;
        if (is_bwd_w) return DNNL_ARG_DIFF_DST;
        ir_error_not_expected();
        return -1;
    }

    bool is_dst_input() const { return is_bwd_d || is_bwd_w; }
    bool is_dst_output() const { return is_fwd; }

    static std::string prepend_groups_to_tag(const std::string &tag) {
        auto ret = tag;
        for (auto &c : ret) {
            bool is_lower_dim = ('a' <= c && c < 'a' + DNNL_MAX_NDIMS);
            bool is_upper_dim = ('A' <= c && c < 'A' + DNNL_MAX_NDIMS);
            if (!is_lower_dim && !is_upper_dim) continue;
            c += 1;
        }
        return "a" + ret;
    }

    static layout_t init_layout(memory_desc_t &md, const std::string &tag) {
        if (md.format_kind != format_kind::any) return make_layout(md);
        auto ret = make_layout(md, tag);
        md = ret.to_dnnl(md.dims);
        return ret;
    }

    static layout_t make_layout(const memory_desc_t &md) {
        return layout_t(md, /*do_normalize=*/false);
    }

    static layout_t make_layout(
            const memory_desc_t &md, const std::string &tag) {
        return layout_t(md, tag, /*do_normalize=*/false);
    }

    static bool with_sum_post_op(const convolution_pd_t *pd) {
        auto &post_ops = pd->attr()->post_ops_;
        return post_ops.find(primitive_kind::sum) != -1;
    }

    static bool is_nhwc(const std::string &tag, const convolution_pd_t *pd,
            const layout_t &layout) {
        const memory_desc_t *md = nullptr;
        if (tag == "src") {
            md = pd->invariant_src_md();
        } else if (tag == "dst") {
            md = pd->invariant_dst_md();
        } else {
            ir_error_not_expected();
        }

        if (!memory_desc_wrapper(md).is_plain()) return false;
        if (layout != make_layout(*md, "axb")) return false;

        return true;
    }

    bool is_nhwc(const std::string &tag, const convolution_pd_t *pd) {
        auto &layout = tensor_config.user_layout(tag);
        return is_nhwc(tag, pd, layout);
    }
};

inline std::ostream &operator<<(std::ostream &out, const conv_config_t &cfg) {
    out << cfg.str();
    return out;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
