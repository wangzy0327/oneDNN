find_path(BANG_INCLUDE_DIR  NAMES "cnrt.h"
    HINTS /usr/local/neuware/include/
)

find_library(BANG_LIBRARY cnrt
    HINTS /usr/local/neuware/lib64/
)

if(BANG_INCLUDE_DIR AND BANG_LIBRARY)
    set(BANG_FOUND TRUE)
else()
    message(FATAL_ERROR "cnrt not found")
endif()