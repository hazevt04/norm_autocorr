cmake_minimum_required(VERSION 3.9)

project(my_cufft_utils VERSION 1.0.0 DESCRIPTION "my_cufft_utils description")

include(GNUInstallDirs)

find_package(MyUtils REQUIRED)
find_package(MyCudaUtils REQUIRED)

# Needed for the CUDA_INCLUDE_DIRCUDS.
# Also needed to enforce the requirement for CUDA
find_package(CUDA REQUIRED)

set(MY_LIB_NAME ${CMAKE_PROJECT_NAME})
set(MY_CONFIG_NAME "MyCufftUtilsConfig")

set(MY_CUFFT_UTILS_LIB_SRCS
    "src/my_cufft_utils.cpp"
    )

set(MY_CUFFT_UTILS_LIB_HEADERS
    "include/my_cufft_utils.hpp"
)

set(MY_CUFFT_UTILS_LIB_INCLUDEDIRS
    "${CUDA_INCLUDE_DIRS}"
)

add_library(${MY_LIB_NAME} SHARED "${MY_CUFFT_UTILS_LIB_SRCS}")

set_target_properties(${MY_LIB_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

target_include_directories(${MY_LIB_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${MY_LIB_NAME}>  # <prefix>/include/mylib
)

target_include_directories(${MY_LIB_NAME} PRIVATE
  "${MY_CUFFT_UTILS_LIB_INCLUDEDIRS}"
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


