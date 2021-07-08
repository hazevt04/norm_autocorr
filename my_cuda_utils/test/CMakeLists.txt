cmake_minimum_required(VERSION 3.8)

project( test_my_cuda_utils
  VERSION 1.0
  LANGUAGES CXX CUDA 
)

set(SRCS
  "src/main.cu"
  )

find_package(CUDA REQUIRED)
find_package(MyUtils REQUIRED)
find_package(MyCudaUtils REQUIRED)

add_executable( ${CMAKE_PROJECT_NAME} "${SRCS}" )

target_include_directories( ${CMAKE_PROJECT_NAME} PRIVATE
  ${MY_CUDA_UTILS_LIB_INCLUDEDIRS}
  ${MY_UTILS_LIB_INCLUDEDIRS}
  ${CUDA_INCLUDE_DIRS}
)

target_link_libraries( ${CMAKE_PROJECT_NAME} INTERFACE ${CUDA_cudadevrt_LIBRARY} ${CUDA_LIBRARY} my_utils)
