# set_source_files_properties(matmul_allreduce.cpp PROPERTIES LANGUAGE ASCEND)
# catlass_example_add_executable(41_matmul_allreduce dav-c220 matmul_allreduce.cpp)

########

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

IF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    SET(ARCH aarch64)
ELSEIF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    SET(ARCH x86_64)
ENDIF ()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

# shmem库编译-device能力
set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend910B2" CACHE STRING "system on chip type")

# 遗留问题: 此处一定要选Debug?
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()

set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}" CACHE STRING "ASCEND_CANN_PACKAGE_PATH")
set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)

# include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# 设置共享内存库的路径
# set(SHMEM_PATH /home/workspace/cdx/luluteam_shmem/shmem)
get_filename_component(SHMEM_PATH $ENV{SHMEM_HOME_PATH} DIRECTORY)
# message(STATUS "SHMEM_PATH: ${SHMEM_PATH}")

set(MPI_HOME /usr/local/mpich)

find_package(MPI REQUIRED)
if (MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
else (MPI_FOUND)
    message(SEND_ERROR "This project requires MPI. Please install MPI first.")
endif (MPI_FOUND)

include_directories(${SHMEM_PATH}/install/memfabric_hybrid/include)
include_directories(${SHMEM_PATH}/install/shmem/include)
include_directories(${SHMEM_PATH}/install/shmem/include/host)
include_directories(${SHMEM_PATH}/install/shmem/include/device)
include_directories(${SHMEM_PATH}/install/shmem/include)
include_directories(${SHMEM_PATH}/install/memfabric_hybrid/include/smem/device)

include_directories(${ASCEND_HOME_PATH}/include)

include_directories(${MPI_HOME}/include)

link_directories(${SHMEM_PATH}/install/shmem/lib)
link_directories(${SHMEM_PATH}/install/memfabric_hybrid/lib)
link_directories(${ASCEND_HOME_PATH}/lib64)

set_source_files_properties(matmul_allreduce.cpp PROPERTIES LANGUAGE ASCEND)
catlass_example_add_executable(41_matmul_allreduce mix matmul_allreduce.cpp)

target_link_libraries(41_matmul_allreduce PRIVATE ascendcl shmem mf_smem mf_hybm_core nnopbase ${MPI_CXX_LIBRARIES})
