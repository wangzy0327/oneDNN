file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

set(OBJ_LIB ${DNNL_LIBRARY_NAME}_sycl_amd)
add_library(${OBJ_LIB} OBJECT ${SOURCES})

find_package(OpenCL REQUIRED)

set_target_properties(
    ${OBJ_LIB}
    PROPERTIES
        COMPILE_DEFINITIONS
        "$<TARGET_PROPERTY:rocBLAS::rocBLAS,INTERFACE_COMPILE_DEFINITIONS>;$<TARGET_PROPERTY:MIOpen::MIOpen,INTERFACE_COMPILE_DEFINITIONS>"
        COMPILE_OPTIONS
        "$<TARGET_PROPERTY:rocBLAS::rocBLAS,INTERFACE_COMPILE_OPTIONS>;$<TARGET_PROPERTY:OpenCL::OpenCL,INTERFACE_COMPILE_OPTIONS>;$<TARGET_PROPERTY:MIOpen::MIOpen,INTERFACE_COMPILE_OPTIONS>"
)
target_include_directories(
    ${OBJ_LIB}
    PRIVATE $<TARGET_PROPERTY:OpenCL::OpenCL,INTERFACE_INCLUDE_DIRECTORIES>
            /opt/rocm/include)

add_library(${OBJ_LIB}_interface INTERFACE)
target_link_libraries(${OBJ_LIB}_interface INTERFACE rocBLAS::rocBLAS
                                                        MIOpen::MIOpen
                                                        OpenCL::OpenCL)
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)

set(${DNNL_LIBRARY_NAME}_INTERFACE
    ${${DNNL_LIBRARY_NAME}_INTERFACE} ${OBJ_LIB}_interface
    PARENT_SCOPE)
