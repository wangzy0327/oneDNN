/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include <assert.h>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::types;

status_t dnnl_matmul_desc_init(matmul_desc_t *matmul_desc,
        const memory_desc_t *src_md, const memory_desc_t *weights_md,
        const memory_desc_t *bias_md, const memory_desc_t *dst_md) {
    bool args_ok = !any_null(matmul_desc, src_md, weights_md, dst_md);
    if (!args_ok) return status::invalid_arguments;

    auto op_d = matmul_desc_t();
    op_d.primitive_kind = primitive_kind::matmul;

    op_d.src_desc = *src_md;
    op_d.weights_desc = *weights_md;
    if (bias_md) op_d.bias_desc = *bias_md;
    op_d.dst_desc = *dst_md;

    const bool with_bias = op_d.bias_desc.ndims != 0;
    const int ndims = dst_md->ndims;
    
    // no check

    op_d.accum_data_type = types::default_accum_data_type(src_md->data_type,
            weights_md->data_type, dst_md->data_type, prop_kind::forward);
    if (op_d.accum_data_type == data_type::undef)
        return status::invalid_arguments;

    *matmul_desc = op_d;
    return status::success;
}
