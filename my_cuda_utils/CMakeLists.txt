cmake_minimum_required(VERSION 3.9)

project(my_cuda_utils VERSION 1.0.0 DESCRIPTION "my_cuda_utils description")

include(GNUInstallDirs)

# Needed for the CUDA_INCLUDE_DIRS.
# Also needed to enforce the requirement for CUDA
find_package(CUDA REQUIRED)

set(MY_LIB_NAME "${CMAKE_PROJECT_NAME}")
set(MY_CONFIG_NAME "MyCudaUtilsConfig")

set(MY_CUDA_UTILS_LIB_HEADERS
	"include/my_cuda_utils.hpp"
	"include/man_vec_file_io_funcs.hpp"
	"include/device_allocator.hpp"
	"include/pinned_vec_file_io_funcs.hpp"
	"include/pinned_mapped_vec_file_io_funcs.hpp"
	"include/pinned_mapped_allocator.hpp"
	"include/pinned_allocator.hpp"
	"include/managed_allocator_host.hpp"
	"include/managed_allocator_global.hpp"
)

set(MY_CUDA_UTILS_LIB_INCLUDEDIRS
    "include"
    "${CUDA_INCLUDE_DIRS}"
)

add_library(${MY_LIB_NAME} INTERFACE)

install(TARGETS ${PROJECT_NAME} EXPORT ${MY_CONFIG_NAME}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(EXPORT ${MY_CONFIG_NAME} DESTINATION share/${MY_CONFIG_NAME}/cmake)

export(TARGETS ${PROJECT_NAME} FILE ${MY_CONFIG_NAME}.cmake)

configure_file(${MY_LIB_NAME}.pc.in ${MY_LIB_NAME}.pc @ONLY)

install(FILES ${CMAKE_BINARY_DIR}/${MY_LIB_NAME}.pc
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
