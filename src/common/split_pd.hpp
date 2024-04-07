/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef COMMON_SPLIT_PD_HPP
#define COMMON_SPLIT_PD_HPP

#include <assert.h>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {

struct split_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::split;

    typedef split_pd_t base_class;
    typedef split_pd_t hint_class;

    const split_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }
    
    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::split_d:
                *(const split_desc_t **)result = desc();
                break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    arg_usage_t arg_usage(int arg) const override {
        if (arg >= DNNL_ARG_MULTIPLE_DST
                && arg < DNNL_ARG_MULTIPLE_DST + n_outputs())
            return arg_usage_t::output;

        if (arg == DNNL_ARG_SRC) return arg_usage_t::input;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(int arg) const override {
        int dst_index = arg - DNNL_ARG_MULTIPLE_DST;
        if (dst_index >= 0 && dst_index < n_outputs()) return dst_md(dst_index);
        if (arg == DNNL_ARG_SRC) return src_md(0);
        return primitive_desc_t::arg_md(arg);
    }

    const memory_desc_t *dst_md(int index = 0) const override {
        return index < n_outputs() ? &dst_mds_[index] : &glob_zero_md;
    }
    
    const memory_desc_t *src_md(int index = 0) const override {
        return index == 0 ? &src_md_ : &glob_zero_md;
    }

    int n_inputs() const override { return 1; }
    int n_outputs() const override { return n_; }

    int split_dim() const { return split_dim_; }

    // const memory_desc_t *dst_image_md(int index = 0) const {
    //     return index < n_outputs() ? &dst_image_mds_[index] : &glob_zero_md;
    // }

protected:
    int n_; // number of dsts
    int split_dim_;
    memory_desc_t src_md_;
    // memory_desc_t original_src_;
    std::vector<memory_desc_t> dst_mds_;

    /* contains images of dsts in the src memory (if possible)
     * Lives here to simplify some implementations. An implementation might
     * use this auxiliary array iff init() returned success */
    // std::vector<memory_desc_t> dst_image_mds_;

protected:
    split_desc_t desc_;
    
    split_pd_t(const split_desc_t *adesc, const primitive_attr_t *attr,
            const split_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind) {
            desc_ = *adesc;
            n_ = desc_.n;
            split_dim_ = desc_.split_dimension;
            src_md_ = desc_.src_md;
            dst_mds_.reserve(n_);
            for (int i = 0; i < n_; ++i)
                dst_mds_.push_back(desc_.dst_mds[i]); 
        }

/*     split_pd_t(const primitive_attr_t *attr, const memory_desc_t *src_md,
            int n, int split_dim, const memory_desc_t *dst_mds)
        : primitive_desc_t(attr, primitive_kind::split)
        , n_(n)
        , split_dim_(split_dim)
        , src_md_(*src_md)
        , original_src_(*src_md) {
        dst_mds_.reserve(n_);
        for (int i = 0; i < n_; ++i)
            dst_mds_.push_back(dst_mds[i]);

        init_desc();
    }

    split_pd_t(const split_pd_t &other) : primitive_desc_t(other) {
        n_ = other.n_;
        split_dim_ = other.split_dim_;
        src_md_ = other.src_md_;
        original_src_ = other.original_src_;
        dst_mds_ = other.dst_mds_;
        dst_image_mds_ = other.dst_image_mds_;

        init_desc();
    } */

    /* inits src_image_mds_ and dst_md_ in simple cases. It is possible to
     * override dst_md_ by using force_src_md.
     * Rationale: if user forces particular dst_md, that cannot be used to
     *            create src_img_mds, the implementation might need to use
     *            intermediate (force_src_md) memory with some plain format.
     *
     * @warning The call may fail. */
    status_t init(const memory_desc_t *force_src_md = nullptr) {
        bool ok = attr()->has_default_values();
        
        // if (force_src_md == nullptr)
        //     ok = ok && set_default_params() == status::success;

        if (!ok) return status::unimplemented;

        /* work with force_src_md */
        if (force_src_md == nullptr) force_src_md = &src_md_;

        // for (int i = 0; i < n_; ++i) {
        //     const memory_desc_wrapper i_d(&src_mds_[i]);
        //     if (!i_d.is_blocking_desc() || i_d.is_additional_buffer())
        //         return status::unimplemented;
        // }

        // const int ndims = force_src_md->ndims;
        // int current_split_dim_offset = 0;
        // for (int i = 0; i < n_; ++i) {
        //     const int dim = src_mds_[i].dims[split_dim_];
        //     dims_t dims, offsets = {};
        //     utils::array_copy(dims, force_src_md->dims, ndims);
        //     dims[split_dim_] = dim;
        //     offsets[split_dim_] = current_split_dim_offset;

        //     memory_desc_t src_img_d;
        //     status_t status = dnnl_memory_desc_init_submemory(
        //             &src_img_d, force_src_md, dims, offsets);
        //     if (status != status::success) {
        //         src_image_mds_.clear();
        //         return status;
        //     }
        //     src_image_mds_.push_back(src_img_d);
        //     current_split_dim_offset += dim;
        // }

        return status::success;
    }

    // useless
    status_t set_default_params() {
        if (src_md_.format_kind != format_kind::any) return status::success;

        // const int ndims = src_md_.ndims;

        // what is this ?
        /* The stupidest ever heuristics (but not the same as we had before):
         *  - Pick the first non-plain format;
         *  - If all formats are plain or it is not possible to create a
         *    blocked format for the output, pick the format of the plain input
         *  - If this fails as well, use plain layout (abcd...)
         */
        status_t status = status::unimplemented;
        for (int i = 0; i < n_; ++i) {
            const memory_desc_wrapper dst_d(dst_mds_[i]);
            if (dst_d.is_blocking_desc() && !dst_d.is_plain()) {
                status = memory_desc_init_by_blocking_desc(
                        src_md_, dst_d.blocking_desc());
                if (status == status::success) break;
            }
        }

        // // i think this is not necessary
        // if (status == status::success) {
        //     /* check if we can create a sub-memory for the dst */
        //     bool desired_format_ok = true;
        //     int current_split_dim_offset = 0;
        //     for (int i = 0; i < n_; ++i) {
        //         const int dim = src_mds_[i].dims[split_dim_];
        //         dims_t dims, offsets = {};
        //         utils::array_copy(dims, dst_md_.dims, ndims);
        //         dims[split_dim_] = dim;
        //         offsets[split_dim_] = current_split_dim_offset;

        //         memory_desc_t src_img_d;
        //         status_t status = dnnl_memory_desc_init_submemory(
        //                 &src_img_d, &dst_md_, dims, offsets);
        //         if (status != status::success) {
        //             desired_format_ok = false;
        //             break;
        //         }
        //         current_split_dim_offset += dim;
        //     }

        //     if (!desired_format_ok) status = status::unimplemented;
        // }

        /* if no success so far, try using the format of the first plain input */
        // if (status != status::success) {
        //     for (int i = 0; i < n_; ++i) {
        //         const memory_desc_wrapper src_d(src_mds_[i]);
        //         if (src_d.is_blocking_desc() && src_d.is_plain()
        //                 && src_d.nelems() > 0) {
        //             status = memory_desc_init_by_blocking_desc(dst_md_,
        //                     memory_desc_wrapper(src_mds_[i]).blocking_desc());
        //             if (status == status::success) return status;
        //         }
        //     }
        // }

        // /* the last line of defense: use plain abcd... format */
        // if (status != status::success)
        //     status = memory_desc_init_by_strides(dst_md_, nullptr);

        return status;
    }

private:
/*     void init_desc() {
        desc_ = split_desc_t();
        desc_.primitive_kind = primitive_kind::split;
        desc_.src_md = &original_src_;
        desc_.n = n_;
        desc_.split_dimension = split_dim_;
        desc_.dst_mds = dst_mds_.data();
    } */
};

} // namespace impl
} // namespace dnnl

#endif
