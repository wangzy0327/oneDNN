/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#include "gpu/cambricon/cnnl_sum.hpp"
#include "gpu/cambricon/sycl_bang_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace cambricon {

namespace {

constexpr impl_list_item_t bang_sum_impl_list[] = {
        impl_list_item_t::sum_type_deduction_helper_t<cnnl_ref_sum_t::pd_t>(),
        nullptr};
} // namespace

const impl_list_item_t *
bang_gpu_engine_impl_list_t::get_sum_implementation_list() {
    return bang_sum_impl_list;
}

} // namespace cambricon
} // namespace gpu
} // namespace impl
} // namespace dnnl
