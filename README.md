编译命令如下：


```shell
# llvm

source pre.sh # 必须指定编译器为clang，gcc不行
mkdir build
cd build
# cmake .. -DDNNL_CPU_RUNTIME=NONE -DDNNL_GPU_RUNTIME=DPCPP -DDNNL_GPU_VENDOR=CAMBRICON -DCMAKE_BUILD_TYPE=Debug -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF
cmake .. -DCMAKE_C_COMPILER=/home/wzy/repos/llvm-mlu/build/bin/clang -DCMAKE_CXX_COMPILER=/home/wzy/repos/llvm-mlu/build/bin/clang++ -DDNNL_CPU_RUNTIME=NONE -DDNNL_GPU_RUNTIME=DPCPP -DDNNL_GPU_VENDOR=CAMBRICON -DCMAKE_BUILD_TYPE=Debug -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/home/wzy/oneDNN-mlu/
make -j
make install
```