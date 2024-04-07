find_path(HIP_INCLUDE_DIR  NAMES "hip/hip_runtime.h"
    HINTS /opt/rocm/include/
)

find_library(HIP_LIBRARY amdhip64
    HINTS /opt/rocm/lib
)

if(HIP_INCLUDE_DIR AND HIP_LIBRARY)
    set(HIP_FOUND TRUE)
else()
    message(FATAL_ERROR "hip not found")
endif()