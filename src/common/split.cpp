/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
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
using namespace dnnl::impl::status;

status_t dnnl_split_desc_init(dnnl_split_desc_t *split_desc, const memory_desc_t *src_md,
                     int n, int split_dim, const memory_desc_t *dst_mds)
{
    bool args_ok = true && !any_null(split_desc, src_md, dst_mds);
    if (!args_ok) return invalid_arguments;

    auto temp_desc = split_desc_t();

	temp_desc.split_dimension = split_dim;
	temp_desc.n = n;
	temp_desc.primitive_kind = primitive_kind::split;
	temp_desc.src_md = *src_md;
	temp_desc.dst_mds = (dnnl_memory_desc_t *)std::malloc(sizeof(dnnl_memory_desc_t)*n);
	for(int i=0; i<n; i++)
		temp_desc.dst_mds[i] = dst_mds[i];

    *split_desc = temp_desc;
    return success;
}

/* namespace dnnl {
namespace impl {

status_t split_primitive_desc_create(primitive_desc_iface_t **split_pd_iface,
        const memory_desc_t *src_md, int n, int split_dim,
        const memory_desc_t *dst_mds, const primitive_attr_t *attr,
        engine_t *engine) {
    if (attr == nullptr) attr = &default_attr();

    dnnl_split_desc_t desc
            = {primitive_kind::split, src_md, n, split_dim, dst_mds};
    primitive_hashing::key_t key(
            engine, reinterpret_cast<op_desc_t *>(&desc), attr, 0, {});
    auto pd = primitive_cache().get_pd(key);

    if (pd) {
        return safe_ptr_assign(
                *split_pd_iface, new primitive_desc_iface_t(pd, engine));
    }

    split_pd_t *split_pd = nullptr;
    for (auto c = engine->get_split_implementation_list(); *c; ++c) {
        if ((*c)(&split_pd, engine, attr, dst_md, n, split_dim, src_mds)
                == success) {
            pd.reset(split_pd);
            CHECK(safe_ptr_assign(
                    *split_pd_iface, new primitive_desc_iface_t(pd, engine)));
            return status::success;
        }
    }
    return unimplemented;
}

} // namespace impl
} // namespace dnnl
 */
/* status_t dnnl_split_primitive_desc_create(
        primitive_desc_iface_t **split_pd_iface, const memory_desc_t *src_md,
        int n, int split_dim, const memory_desc_t *dst_mds,
        const primitive_attr_t *attr, engine_t *engine) {
    return split_primitive_desc_create(
            split_pd_iface, src_md, n, split_dim, dst_mds, attr, engine);
} */
