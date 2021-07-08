cmake_minimum_required( VERSION 3.9 )

project( norm_autocorr 
   VERSION 1.0.0 
   LANGUAGES CXX CUDA
   DESCRIPTION "Normalized Autocorrelation library" )

include(GNUInstallDirs)

find_package(MyUtils REQUIRED)
find_package(MyCudaUtils REQUIRED)
find_package(MyCufftUtils REQUIRED)

set(CUDA_VERBOSE_BUILD ON)
find_package(CUDA REQUIRED)

set( MY_LIB_NAME "${CMAKE_PROJECT_NAME}" )
set( MY_CONFIG_NAME "NormAutocorrConfig" )

set( NORM_AUTOCORR_LIB_SRCS
  "src/my_args.cpp"
  "src/norm_autocorr_kernel.cu"
  "src/NormAutocorrGPU.cu"
)

set( NORM_AUTOCORR_LIB_HEADERS
  "include/my_args.hpp"
  "include/norm_autocorr_kernel.cuh"
  "include/NormAutocorrGPU.hpp"
)

set(NORM_AUTOCORR_LIB_INCLUDEDIRS
    "${CUDA_INCLUDE_DIRS}"
)

cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)

LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
LIST(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
LIST(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

add_library(${MY_LIB_NAME} SHARED "${NORM_AUTOCORR_LIB_SRCS}")

set_target_properties(${MY_LIB_NAME} PROPERTIES
   VERSION ${PROJECT_VERSION}
   SOVERSION 1
   CXX_STANDARD 14
   CXX_STANDARD_REQUIRED YES
   CXX_EXTENSIONS NO
   CUDA_SEPARABLE_COMPILATION ON
   CMAKE_CUDA_STANDARD 11
   CMAKE_CUDA_STANDARD_REQUIRED ON
)
target_include_directories(${MY_LIB_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${MY_LIB_NAME}>  # <prefix>/include/mylib
)

target_include_directories(${MY_LIB_NAME} PRIVATE
   "${NORM_AUTOCORR_LIB_INCLUDEDIRS}"
)

install(TARGETS ${PROJECT_NAME} EXPORT ${MY_CONFIG_NAME}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(EXPORT ${MY_CONFIG_NAME} DESTINATION share/${MY_CONFIG_NAME}/cmake)


configure_file(${MY_LIB_NAME}.pc.in ${MY_LIB_NAME}.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/${MY_LIB_NAME}.pc
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

