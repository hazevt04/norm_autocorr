cmake_minimum_required(VERSION 3.8)

project( test_my_utils
  VERSION 1.0
  LANGUAGES CXX 
)

set(SRCS
  "src/main.cpp"
  )

find_package(MyUtils REQUIRED)

add_executable( ${CMAKE_PROJECT_NAME} "${SRCS}" )

target_link_libraries( ${CMAKE_PROJECT_NAME} PRIVATE my_utils )
