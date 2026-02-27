# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

option(COMPILE_DYNAMIC_OPTIMIZED_MATMUL "Generate wrapper code for generalization matmul" OFF)

if(COMPILE_DYNAMIC_OPTIMIZED_MATMUL)

    if(NOT DEFINED Python3_EXECUTABLE)
        message(WARNING "Python3_EXECUTABLE is not defined, using default python3")
        find_program(Python3_EXECUTABLE python3 python)
        if(NOT Python3_EXECUTABLE)
            message(FATAL_ERROR "Python executable not found. Please install Python 3.")
        endif()
    else()
        message("Using python: ${Python3_EXECUTABLE}")
    endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts/wrapper_code_gen.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts
    )

    add_custom_target(auto_gen_code)
    file(GLOB_RECURSE AUTO_GEN_CODE_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts/*.py)
    add_custom_command(
        TARGET auto_gen_code
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts/wrapper_code_gen.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/impl/scripts
        DEPENDS ${AUTO_GEN_CODE_SRC}
        COMMENT "Generate wrapper code, launch_map.h ...done"
    )
    
    add_compile_definitions(TILING_KEY_VAR)
    add_compile_options($<$<COMPILE_LANGUAGE:ASCEND>:--cce-aicore-arch=dav-c220>)
    file(GLOB CATLASS_SHARED_LIB_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/impl/wrapper/*.cpp)
    set_source_files_properties(${CATLASS_SHARED_LIB_SRC} PROPERTIES LANGUAGE ASCEND)

    add_library(dynamic_optimized_kernel STATIC ${CATLASS_SHARED_LIB_SRC})
    set_target_properties(dynamic_optimized_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)

    include_directories(
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/include/experiment/runtime
        ${ASCEND_HOME_PATH}/include/experiment/msprof
        ${ASCEND_HOME_PATH}/include/aclnn
        ${CMAKE_CURRENT_SOURCE_DIR}/impl
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

    link_libraries(dynamic_optimized_kernel runtime)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)

    add_executable(102_dynamic_optimized_matmul dynamic_optimized_matmul.cpp)
    add_dependencies(catlass_examples 102_dynamic_optimized_matmul)
    add_dependencies(102_dynamic_optimized_matmul dynamic_optimized_kernel)
    add_dependencies(dynamic_optimized_kernel auto_gen_code)

    install(TARGETS dynamic_optimized_kernel DESTINATION shared_lib/lib COMPONENT 102_dynamic_optimized_matmul)
    install(TARGETS dynamic_optimized_kernel DESTINATION shared_lib/lib COMPONENT catlass_examples)
    install(TARGETS 102_dynamic_optimized_matmul DESTINATION bin COMPONENT 102_dynamic_optimized_matmul)
    install(TARGETS 102_dynamic_optimized_matmul DESTINATION bin COMPONENT catlass_examples)

endif()