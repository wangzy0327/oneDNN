find_package(Threads REQUIRED)

find_package(HIP)

find_path(MIOPEN_INCLUDE_DIR miopen/miopen.h 
    HINTS /opt/rocm/include
)
find_library(MIOPEN_LIBRARY MIOpen
    HINTS /opt/rocm/lib
)

if(MIOPEN_INCLUDE_DIR AND MIOPEN_LIBRARY)
    set(MIOPEN_FOUND TRUE)
else()
    message(FATAL_ERROR "miopen not found!")
endif()

# TODO version adjustment, or maybe no need to

if(NOT TARGET MIOpen::MIOpen)
    add_library(MIOpen::MIOpen SHARED IMPORTED)
    target_compile_definitions(MIOpen::MIOpen INTERFACE __HIP_PLATFORM_AMD__)
    set_target_properties(MIOpen::MIOpen PROPERTIES
        INTERFACE_LINK_LIBRARIES
            "Threads::Threads;${HIP_LIBRARY}"
        IMPORTED_LOCATION 
            ${MIOPEN_LIBRARY}
	)
endif()