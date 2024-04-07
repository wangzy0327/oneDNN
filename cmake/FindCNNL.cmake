find_package(Threads REQUIRED)

find_package(BANG)

find_path(CNNL_INCLUDE_DIR cnnl.h
    HINTS /usr/local/neuware/include/
)
find_library(CNNL_LIBRARY cnnl
    HINTS /usr/local/neuware/lib64
)

if(CNNL_INCLUDE_DIR AND CNNL_LIBRARY)
    set(CNNL_FOUND TRUE)
else()
    message(FATAL_ERROR "cnnl not found!")
endif()

# TODO version adjustment, or maybe no need to

if(NOT TARGET cnnl::cnnl)
    add_library(cnnl::cnnl SHARED IMPORTED)
    set_target_properties(cnnl::cnnl PROPERTIES
        INTERFACE_LINK_LIBRARIES
            "Threads::Threads;${BANG_LIBRARY}"
        IMPORTED_LOCATION 
            ${CNNL_LIBRARY}
	)
endif()