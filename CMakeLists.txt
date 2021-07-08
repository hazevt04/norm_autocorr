cmake_minimum_required( VERSION 3.8 )

project( norm_autocorr 
   VERSION 1.0.0 
   LANGUAGES CXX CUDA
   DESCRIPTION "Normalized Autocorrelation library" )

include(GNUInstallDirs)

set( MY_LIB_NAME "${CMAKE_PROJECT_NAME}" )

set( MY_LIB_SRCS
   "src/norm_autocorr_kernel.cu"
   "src/NormAutocorrGPU.cu"
)

set( MY_LIB_HEADERS
   "include/norm_autocorr_kernel.cuh"
   "include/NormAutocorrGPU.hpp"
)

set(MY_LIB_INCLUDEDIRS
    "include"
    "${CUDA_INCLUDE_DIRS}"
)

set(CUDA_VERBOSE_BUILD ON)

find_package(CUDA REQUIRED)
cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)

LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
LIST(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
LIST(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

add_library(${MY_LIB_NAME} SHARED "${MY_LIB_SRCS}")

set_target_properties(${MY_LIB_NAME} PROPERTIES
   VERSION ${PROJECT_VERSION}
   SOVERSION 1
   PUBLIC_HEADER "${MY_LIB_HEADERS}"
   CXX_STANDARD 14
   CXX_STANDARD_REQUIRED YES
   CXX_EXTENSIONS NO
   CUDA_SEPARABLE_COMPILATION ON
   CMAKE_CUDA_STANDARD 11
   CMAKE_CUDA_STANDARD_REQUIRED ON
)

target_include_directories(${MY_LIB_NAME} PRIVATE "${MY_LIB_INCLUDEDIRS}")


configure_file(${MY_LIB_NAME}.pc.in ${MY_LIB_NAME}.pc @ONLY)

install(TARGETS ${MY_LIB_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(FILES ${CMAKE_BINARY_DIR}/${MY_LIB_NAME}.pc
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

