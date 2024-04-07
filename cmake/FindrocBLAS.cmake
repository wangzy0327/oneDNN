find_package(Threads REQUIRED)

find_package(HIP)

find_library(ROCBLAS_LIBRARY rocblas
    HINTS /opt/rocm/lib
)

# TODO version adjustment

if(NOT TARGET rocBLAS::rocBLAS)
    add_library(rocBLAS::rocBLAS SHARED IMPORTED)
    target_compile_definitions(rocBLAS::rocBLAS INTERFACE __HIP_PLATFORM_AMD__)
    # 下面的语句不要把rocblas的头文件(include路径)也加进来
    set_target_properties(rocBLAS::rocBLAS PROPERTIES
        INTERFACE_LINK_LIBRARIES
            "Threads::Threads;${HIP_LIBRARY}"
        IMPORTED_LOCATION 
            ${ROCBLAS_LIBRARY}
	)
endif()

# if(NOT TARGET rocBLAS::rocBLAS)
#     add_library(rocBLAS::rocBLAS SHARED IMPORTED)
#     target_link_libraries(rocBLAS::rocBLAS
#         INTERFACE "Threads::Threads"
#         INTERFACE /opt/rocm/hip/lib/libamdhip64.so
#         INTERFACE /opt/rocm/rocblas/lib/librocblas.so
#     )
#     target_compile_definitions(rocBLAS::rocBLAS INTERFACE __HIP_PLATFORM_AMD__)
#     set_target_properties(rocBLAS::rocBLAS
#         PROPERTIES
#         IMPORTED_LOCATION /opt/rocm/rocblas/lib/librocblas.so
# 	)
    
# endif()