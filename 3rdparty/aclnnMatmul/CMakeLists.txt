# CMake lowest version requirement
cmake_minimum_required(VERSION 3.14)

# 设置工程名
project(ACLNN_MATMUL_EXAMPLE)

# Compile options
add_compile_options(-std=c++11)

# 设置编译选项
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./bin")    
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# 设置可执行文件名为 aclmatmul_test，并指定待运行算子文件为 aclmatmul.cpp
add_executable(aclmatmul_test
               aclmatmul.cpp) 

# 设置ASCEND_PATH（CANN软件包目录，请根据实际路径修改）和INCLUDE_BASE_DIR（头文件目录）
if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")      
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()
set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
include_directories(
    ${INCLUDE_BASE_DIR}
    ${INCLUDE_BASE_DIR}/aclnn
)

# 设置链接的库文件路径
target_link_libraries(aclmatmul_test PRIVATE
                      ${ASCEND_PATH}/lib64/libascendcl.so
                      ${ASCEND_PATH}/lib64/libnnopbase.so
                      ${ASCEND_PATH}/lib64/libopapi.so)

# 可执行文件在 CMakeLists 文件所在目录的 bin 目录下
install(TARGETS aclmatmul_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
