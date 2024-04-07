#===============================================================================
# Copyright 2019-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

if(SYCL_cmake_included)
    return()
endif()
set(SYCL_cmake_included true)

if(NOT DNNL_WITH_SYCL)
    if (NOT DNNL_DPCPP_HOST_COMPILER STREQUAL "DEFAULT")
        message(FATAL_ERROR "DNNL_DPCPP_HOST_COMPILER is supported only for DPCPP runtime")
    endif()
    return()
endif()

include(FindPackageHandleStandardArgs)
include("cmake/dpcpp_driver_check.cmake")

find_package(LevelZero)
if(LevelZero_FOUND)
    message(STATUS "DPC++ support is enabled (OpenCL and Level Zero)")
else()
    message(STATUS "DPC++ support is enabled (OpenCL)")
endif()

# Explicitly link against sycl as Intel oneAPI DPC++ Compiler does not
# always do it implicitly.
if(WIN32)
    list(APPEND EXTRA_SHARED_LIBS
        $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithMDd>>:sycld>
        $<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<CONFIG:RelWithMDd>>>:sycl>)
else()
    list(APPEND EXTRA_SHARED_LIBS sycl)
endif()

if(NOT DNNL_DPCPP_HOST_COMPILER STREQUAL "DEFAULT" AND (DNNL_SYCL_CUDA OR DNNL_SYCL_HIP OR DNNL_SYCL_BANG))
    message(FATAL_ERROR "DNNL_DPCPP_HOST_COMPILER options is not supported for NVIDIA or AMD or CAMBRICON.")
endif()

if(DNNL_SYCL_CUDA)
    # XXX: Suppress warning coming from SYCL headers:
    #   error: use of function template name with no prior declaration in
    #   function call with eplicit template arguments is a C++20 extension
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++20-extensions")

    # Explicitly linking against OpenCL without finding the right one can
    # end up linking the tests against Nvidia OpenCL. This can be
    # problematic as Intel OpenCL CPU backend will not work. When multiple
    # OpenCL backends are available we need to make sure that we are linking
    # against the correct one.
    find_package(OpenCL REQUIRED)
    find_package(cuBLAS REQUIRED)
    find_package(cuDNN REQUIRED)

    if(NOT WIN32)
        # XXX: CUDA contains OpenCL headers that conflict with the OpenCL
        # headers found via `find_package(OpenCL REQUIRED)` above. The
        # workaround is the following:
        # Get interface include directories from all CUDA related import
        # targets and lower their priority via `-idirafter` so that the
        # compiler picks up the OpenCL headers that have been found via
        # `find_package(OpenCL REQUIRED)` above.
        set(cuda_include_dirs)
        foreach(cuda_import_target cuBLAS::cuBLAS;cuDNN::cuDNN)
            get_target_property(cuda_import_target_include_dirs ${cuda_import_target} INTERFACE_INCLUDE_DIRECTORIES)
            set_target_properties(${cuda_import_target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
            list(APPEND cuda_include_dirs ${cuda_import_target_include_dirs})
        endforeach()

        list(REMOVE_DUPLICATES cuda_include_dirs)
        foreach(cuda_include_dir ${cuda_include_dirs})
            append(CMAKE_CXX_FLAGS "-idirafter${cuda_include_dir}")
        endforeach()
    endif()
    list(APPEND EXTRA_SHARED_LIBS OpenCL::OpenCL)
elseif(DNNL_SYCL_HIP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++20-extensions")
    find_package(OpenCL REQUIRED)
    find_package(rocBLAS REQUIRED)
    find_package(MIOpen REQUIRED)
    # message(FATAL_ERROR "opencl library: ${OpenCL_LIBRARY}")
    if(NOT WIN32)
        set(hip_include_dirs)
        # seems no need?
        foreach(hip_import_target rocBLAS::rocBLAS)
            get_target_property(hip_import_target_include_dirs ${hip_import_target} INTERFACE_INCLUDE_DIRECTORIES)
            set_target_properties(${hip_import_target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
            list(APPEND hip_include_dirs ${hip_import_target_include_dirs})
        endforeach()

        list(REMOVE_DUPLICATES hip_include_dirs)
        foreach(hip_include_dir ${hip_include_dirs})
            append(CMAKE_CXX_FLAGS "-idirafter${hip_include_dirs}")
        endforeach()
    endif()
    list(APPEND EXTRA_SHARED_LIBS OpenCL::OpenCL)
elseif(DNNL_SYCL_BANG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++20-extensions")
    # find_package(OpenCL REQUIRED)
    find_package(CNNL REQUIRED)
    # message(FATAL_ERROR "opencl library: ${OpenCL_LIBRARY}")
    if(NOT WIN32)
        set(bang_include_dirs) # Unset
        # TODO
        foreach(bang_import_target cnnl::cnnl)
            get_target_property(bang_import_target_include_dirs ${bang_import_target} INTERFACE_INCLUDE_DIRECTORIES)
            set_target_properties(${bang_import_target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
            list(APPEND bang_include_dirs ${bang_import_target_include_dirs})
        endforeach()

        list(REMOVE_DUPLICATES bang_include_dirs)
        foreach(bang_include_dir ${bang_include_dirs})
            append(CMAKE_CXX_FLAGS "-idirafter${bang_include_dirs}")
        endforeach()
    endif()
    # list(APPEND EXTRA_SHARED_LIBS OpenCL::OpenCL)

else()
    find_library(OPENCL_LIBRARY OpenCL PATHS ENV LIBRARY_PATH ENV LIB NO_DEFAULT_PATH)
    if(OPENCL_LIBRARY)
        message(STATUS "OpenCL runtime is found in the environment: ${OPENCL_LIBRARY}")
        # OpenCL runtime was found in the environment hence simply add it to
        # the EXTRA_SHARED_LIBS list
        list(APPEND EXTRA_SHARED_LIBS ${OPENCL_LIBRARY})
    else()
        message(STATUS "OpenCL runtime is not found in the environment. Try to find it using find_package(...)")
        # This is expected when using OSS compiler that doesn't distribute
        # OpenCL runtime
        find_package(OpenCL REQUIRED)
        # Unset INTERFACE_INCLUDE_DIRECTORIES property because DPCPP
        # compiler contains OpenCL headers
        set_target_properties(OpenCL::OpenCL PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
        list(APPEND EXTRA_SHARED_LIBS OpenCL::OpenCL)
    endif()
endif()

# XXX: Suppress warning coming from SYCL headers:
#   #pragma message("The Intel extensions have been moved into cl_ext.h.
#   Please include cl_ext.h directly.")
if(NOT WIN32)
    if(${CMAKE_VERSION} VERSION_LESS "3.1.0")
        # Prior to CMake 3.1 the Makefile generators did not escape # correctly
        # inside make variable assignments used in generated makefiles, causing
        # them to be treated as comments. This is a workaround.
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-\\#pragma-messages")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-#pragma-messages")
    endif()
endif()

add_definitions_with_host_compiler("-DCL_TARGET_OPENCL_VERSION=300")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

if(LevelZero_FOUND)
    set(DNNL_WITH_LEVEL_ZERO TRUE)
    include_directories_with_host_compiler(${LevelZero_INCLUDE_DIRS})
endif()
