set(CMAKE_POSITION_INDEPENDENT_CODE ON)

IF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    SET(ARCH aarch64)
ELSEIF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    SET(ARCH x86_64)
ENDIF ()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

# shmem库编译-device能力
set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend910B3" CACHE STRING "system on chip type")

# 遗留问题: 此处一定要选Debug?
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()

set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}" CACHE STRING "ASCEND_CANN_PACKAGE_PATH")
set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)

#include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# 设置共享内存库的路径
#set(SHMEM_PATH ${CMAKE_SOURCE_DIR}/shmem)
# set(SHMEM_PATH /home/workspace/cdx/luluteam_shmem/shmem)
get_filename_component(SHMEM_PATH $ENV{SHMEM_HOME_PATH} DIRECTORY)
# message(STATUS "SHMEM_PATH: ${SHMEM_PATH}")

#include_directories(${SHMEM_PATH}/install/memfabric_hybrid/include)
#include_directories(${SHMEM_PATH}/install/shmem/include)
#include_directories(${SHMEM_PATH}/install/shmem/include/host)
#include_directories(${SHMEM_PATH}/install/shmem/include/device)
#include_directories(${SHMEM_PATH}/install/shmem/include)
#include_directories(${SHMEM_PATH}/install/memfabric_hybrid/include/smem/device)



include_directories(${ASCEND_HOME_PATH}/include)

link_directories(${SHMEM_PATH}/install/shmem/lib)
link_directories(${SHMEM_PATH}/install/memfabric_hybrid/lib)
link_directories(${ASCEND_HOME_PATH}/lib64)

set_source_files_properties(grouped_matmul_alltoallv.cpp PROPERTIES LANGUAGE ASCEND)
catlass_example_add_executable(43_grouped_matmul_alltoallv dav-c220 grouped_matmul_alltoallv.cpp)

target_include_directories(43_grouped_matmul_alltoallv PRIVATE
  ${SHMEM_PATH}/install/memfabric_hybrid/include
  ${SHMEM_PATH}/install/memfabric_hybrid/include/smem/device
  ${SHMEM_PATH}/install/shmem/include
  ${SHMEM_PATH}/install/shmem/include/host
  ${SHMEM_PATH}/install/shmem/include/device
  ${ASCEND_HOME_PATH}/include
)

target_link_libraries(43_grouped_matmul_alltoallv PRIVATE ascendcl shmem_host shmem_device mf_smem mf_hybm_core)