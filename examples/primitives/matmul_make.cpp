/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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


#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <CL/sycl.hpp>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

//compile
// clang++  -fsycl -fsycl-targets=mlisa-cambricon-bang matmul.cpp -o matmul.out -ldnnl
//execuate
// ./matmul.out gpu

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void matmul_example(dnnl::engine::kind engine_kind) {

    // sycl::gpu_selector Selector;
    // sycl::context ctx;
    // sycl::device dev = ctx.get_devices()[0];
    // queue Q(Selector);

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);
    // dnnl::engine make_engine(dev,ctx);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);
    // dnnl::stream make_stream(engine,Q);

    // Tensor dimensions.
    const memory::dim M = 1024, K = 1024, N = 1024;

    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {M, K};
    memory::dims weights_dims = {K, N};
    // memory::dims bias_dims = {1, N};
    memory::dims dst_dims = {M, N};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> weights_data(product(weights_dims));
    // std::vector<float> bias_data(product(bias_dims));
    std::vector<float> dst_data(product(dst_dims));

    // Initialize src, weights, bias.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        // return std::cos(i++ / 10.f);
        return 1.0f;
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        static int i = 0;
        // return std::sin(i++ * 2.f);
        return 1.0f;
    });
    // std::generate(bias_data.begin(), bias_data.end(), []() {
    //     static int i = 0;
    //     // return std::tanh(float(i++));
    //     return 1.0f;
    // });

    sycl::buffer<float, 2> buffer_src(static_cast<float*>(src_data.data()), sycl::range<2>( M, K));
    sycl::buffer<float, 2> buffer_weights(static_cast<float*>(weights_data.data()), sycl::range<2>(K, N));
    // sycl::buffer<float, 2> buffer_bias(static_cast<float*>(bias_data.data()), sycl::range<2>(1, N));
    sycl::buffer<float, 2> buffer_dst(static_cast<float*>(dst_data.data()), sycl::range<2>(M, N)); 

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md = memory::desc(src_dims, dt::f32, tag::ab);
    auto weights_md = memory::desc(weights_dims, dt::f32, tag::ab);
    // auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
    auto dst_md = memory::desc(dst_dims, dt::f32, tag::ab);

    // auto src_mem = memory(src_md, engine);
    // auto weights_mem = memory(weights_md, engine);
    // // // auto bias_mem = memory(bias_md, engine);
    // auto dst_mem = memory(dst_md, engine);

    auto src_mem = dnnl::sycl_interop::make_memory(src_md, engine, buffer_src);
    auto weights_mem = dnnl::sycl_interop::make_memory(weights_md, engine, buffer_weights);
    // auto bias_mem = dnnl::sycl_interop::make_memory(bias_md, engine, buffer_bias);
    auto dst_mem = dnnl::sycl_interop::make_memory(dst_md, engine, buffer_dst);

    // Write data to memory object's handles.
    write_to_dnnl_memory(src_data.data(), src_mem);
    write_to_dnnl_memory(weights_data.data(), weights_mem);
    // write_to_dnnl_memory(bias_data.data(), bias_mem);

    // Create primitive post-ops (ReLU).
    // const float alpha = 0.f;
    // const float beta = 0.f;
    // post_ops matmul_ops;
    // matmul_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
    // primitive_attr matmul_attr;
    // matmul_attr.set_post_ops(matmul_ops);

    // auto matmul_d = matmul::desc(src_md,weights_md,bias_md,dst_md);
    auto matmul_d = matmul::desc(src_md,weights_md,dst_md);
    // Create primitive descriptor.
    // auto matmul_pd = matmul::primitive_desc(
    //         engine, src_md, weights_md, bias_md, dst_md, matmul_attr);
    auto matmul_pd = matmul::primitive_desc(
            matmul_d, engine);

    // Create the primitive.
    auto matmul_prim = matmul(matmul_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, src_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
    // matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
    matmul_args.insert({DNNL_ARG_DST, dst_mem});

    // Primitive execution: matrix multiplication with ReLU.
    matmul_prim.execute(engine_stream, matmul_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), dst_mem);
    int num = std::min(dst_data.size(),static_cast<size_t>(100));
    for(int i = 0;i < num;i++){
        if(i == 0)
            std::cout<<"[ ";
        if(i != num - 1)
            std::cout<<dst_data[i]<<",";
        else if(i == num - 1)
            std::cout<<dst_data[i]<<" ]"<<std::endl;
    }
}

int main(int argc, char **argv) {
    return handle_example_errors(matmul_example, parse_engine_kind(argc, argv));
}