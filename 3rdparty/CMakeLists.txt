cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(synopic-baseline)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

if(NOT DEFINED ENV{ATB_HOME_PATH})
    message(FATAL_ERROR "Cannot find ATB_HOME_PATH, please run set_env.sh.")
else()
    set(ATB_HOME_PATH $ENV{ATB_HOME_PATH})
endif()

find_package(MPI)
if(NOT MPI_FOUND)
    message(FATAL_ERROR "Cannot find MPI, please install it first.")
endif()

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(examples)
