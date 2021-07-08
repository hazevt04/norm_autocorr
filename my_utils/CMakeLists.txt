cmake_minimum_required(VERSION 3.9)

project(my_utils VERSION 1.0.0 DESCRIPTION "my_utils description")

include(GNUInstallDirs)

set(MY_LIB_NAME ${CMAKE_PROJECT_NAME})

set(MY_CONFIG_NAME "MyUtilsConfig")

set(MY_UTILS_LIB_SRCS
    "src/my_utils.cpp"
    "src/my_file_io_funcs.cpp"
    )

set(MY_UTILS_LIB_HEADERS
    "include/my_utils.hpp"
    "include/my_file_io_funcs.hpp"
    )

set(MY_UTILS_LIB_INC_DIR "include")

add_library(${MY_LIB_NAME} SHARED "${MY_UTILS_LIB_SRCS}")

set_target_properties(${MY_LIB_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${MY_UTILS_LIB_HEADERS}")

target_include_directories(${MY_LIB_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${MY_LIB_NAME}>  # <prefix>/include/mylib
)

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
