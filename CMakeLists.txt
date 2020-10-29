make_minimum_required( VERSION 3.8 )

project( norm_autocorr LANGUAGES CXX CUDA )

list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
LIST(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

set(CUDA_VERBOSE_BUILD ON)

set(EXEC_NAME "${CMAKE_PROJECT_NAME}")

set(SRCS
   "src/my_utils.cpp"
   "src/my_file_io_funcs.cpp"
   "src/my_cufft_utils.cpp"
   "src/norm_auto_kernel.cu"
   "src/NormAutoGPU.cu"
   "src/main.cpp"
   )

find_package(CUDA REQUIRED)
include_directories(
   "include"
   "${CUDA_INCLUDE_DIRS}"
)

add_executable( ${EXEC_NAME} "${SRCS}" )
target_link_libraries( ${EXEC_NAME} ${CUDA_cudadevrt_LIBRARY} m )

set_target_properties( ${EXEC_NAME}
   PROPERTIES
   CXX_STANDARD 11
   CXX_STANDARD_REQUIRED YES
   CXX_EXTENSIONS NO
   CUDA_SEPARABLE_COMPILATION ON
   CMAKE_CUDA_STANDARD 11
   CMAKE_CUDA_STANDARD_REQUIRED ON
)
