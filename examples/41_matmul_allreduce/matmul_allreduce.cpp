/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>

#include <mpi.h>

#include "catlass/gemm/kernel/optimized_matmul_tla.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "catlass/distributed/block/block_comm.hpp"
#include "catlass/distributed/dispatch_policy.hpp"
#include "catlass/distributed/kernel/matmul_allreduce.hpp"
#include "catlass/layout/layout.hpp"

#include "golden.hpp"
#include "helper.hpp"

#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_rma.h"
#include "host/shmem_host_team.h"

#include "shmem_api.h"

using namespace Catlass;
using namespace tla;

template <class Layout>
auto GetPaddingLayout(Layout layout, uint32_t blockRows, uint32_t blockCols) {
    if constexpr (std::is_same_v<Layout, layout::RowMajor>) {
        auto shape = MakeShape(
            MakeShape(blockRows, CeilDiv(layout.shape(0), blockRows)),
            MakeShape(blockCols, CeilDiv(layout.shape(1), blockCols))
        );
        auto stride = MakeStride(
            MakeStride(
                static_cast<int64_t>(blockCols), static_cast<int64_t>(blockRows) * RoundUp(layout.shape(1), blockCols)
            ),
            MakeStride(Int<1>{}, static_cast<int64_t>(blockRows) * blockCols)
        );
        return MakeLayout(shape, stride);
    } else {
        auto shape = MakeShape(
            MakeShape(blockRows, CeilDiv(layout.shape(0), blockRows)),
            MakeShape(blockCols, CeilDiv(layout.shape(1), blockCols))
        );
        auto stride = MakeStride(
            MakeStride(Int<1>{}, static_cast<int64_t>(blockRows) * blockCols),
            MakeStride(
                static_cast<int64_t>(blockRows), RoundUp(layout.shape(0), blockRows) * static_cast<int64_t>(blockCols)
            )
        );
        return MakeLayout(shape, stride);
    }
}

using Options = CommAROptions;

template <class Layout>
size_t GetWorkspaceLen(Layout layout, size_t blockRows, size_t blockCols) {
    return RoundUp(static_cast<size_t>(layout.shape(0)), blockRows)
           * RoundUp(static_cast<size_t>(layout.shape(1)), blockCols);
}

void BalanceWorkload(uint32_t m, uint32_t n, uint32_t& m1, uint32_t& n1, uint32_t threshold, uint32_t coreNum)
{
    uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), coreNum);
    while (m1 > threshold && (CeilDiv(m, m1 - 16) * CeilDiv(n, n1) <= maxBlocks)) {
        m1 -= 16;
    }
    if (m < m1) {
        m1 = RoundUp(m, uint32_t(16));
    }
    if (n < n1) {
        n1 = RoundUp(n, uint32_t(16));
    }
}
template <class DType>
bool JudgeSpace(uint32_t m1, uint32_t n1, uint32_t k1)
{
    uint64_t l1Size{512 * 1024};
    uint64_t l0CSize{128 * 1024};
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::L1, l1Size);
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
            platform_ascendc::CoreMemType::L0_C, l0CSize);
    bool judgeL1 = (m1 * k1 * 2 * sizeof(DType) + k1 * n1 * 2 * sizeof(DType) <= l1Size);
    bool judgeL0C = (m1 * n1 * 4 <= l0CSize) ? true : false;
    return judgeL1 && judgeL0C;
}

template <class DType>
uint32_t GetMaxK1(uint32_t m1, uint32_t n1)
{
    std::vector<uint32_t> k1List = {1024, 512, 256, 128};
    uint32_t k1 = 512 / sizeof(DType);
    for (const auto &k1t : k1List) {
        if (JudgeSpace<DType>(m1, n1, k1t)) {
            k1 = k1t;
            break;
        }
    }
    return k1;
}

void DoTilingB16Layout00(const GemmCoord &problemShape, GemmCoord &L1Shape, uint32_t coreNum)
{
    uint32_t m = problemShape.m();
    uint32_t n = problemShape.n();
    uint32_t k = problemShape.k();
    uint32_t m1 = 128, n1 = 256, k1 = 256;

    if (n >= 256) {
        // n0 = 256 delivers optimal bandwidth performance.
        uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), coreNum);
        BalanceWorkload(m, n, m1, n1, 32, coreNum);
        uint32_t blocks = CeilDiv(m, uint32_t(64)) * CeilDiv(n, uint32_t(512));
        if (blocks <= maxBlocks - coreNum && k <= 128) {
            m1 = 64;
            n1 = 512;
        }
    } else {
        m1 = 128;
        n1 = RoundUp(n, uint32_t(16));
        uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), coreNum);
        uint32_t m1t = m1;
        while (JudgeSpace<fp16_t>(m1t + 16, n1, k1)) {
            m1t += 16;
            uint32_t blocks = CeilDiv(m, m1t) * CeilDiv(n, n1);
            if (blocks <= maxBlocks - coreNum) {
                m1 = m1t;
            }
        }
        BalanceWorkload(m, n, m1, n1, 32, coreNum);
    }
    if (k >= 65536 || n >= 65536) {
        m1 = 128;
        n1 = 256;
    }
    k1 = GetMaxK1<fp16_t>(m1, n1);
    L1Shape.m() = m1;
    L1Shape.n() = n1;
    L1Shape.k() = k1;
}

void DoTilingB16Layout01(const GemmCoord &problemShape, GemmCoord &L1Shape, uint32_t coreNum)
{
    uint32_t m = problemShape.m();
    uint32_t n = problemShape.n();
    uint32_t k = problemShape.k();
    uint32_t m1 = 128, n1 = 256, k1 = 256;
    // When LayoutA is RowMajor and LayoutB is ColumnMajor, bandwidth issues can be completely disregarded,
    // simply choose the tiling configureation with the most balanced workload
    double ratio = (double)(m * k + k * n) / (m * n);
    if (m > n && (ratio > 0.1 || n < 256)) {
        m1 = 256;
        n1 = 128;
        BalanceWorkload(m, n, m1, n1, 64, coreNum);
        BalanceWorkload(n, m, n1, m1, 64, coreNum); 
    } else {
        BalanceWorkload(n, m, n1, m1, 64, coreNum);
        BalanceWorkload(m, n, m1, n1, 64, coreNum); 
    }
    uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), coreNum);
    if (m < n) {
        uint32_t n1t = n1;
        while (JudgeSpace<fp16_t>(m1, n1t + 16, k1)) {
            n1t += 16;
            uint32_t blocks = CeilDiv(m, m1) * CeilDiv(n, n1t);
            if (blocks <= maxBlocks - coreNum) {
                n1 = n1t;
            }
        }
        BalanceWorkload(m, n, m1, n1, 64, coreNum);
        BalanceWorkload(n, m, n1, m1, 64, coreNum); 
    } else {
        uint32_t m1t = m1;
        while (JudgeSpace<fp16_t>(m1t + 16, n1, k1)) {
            m1t += 16;
            uint32_t blocks = CeilDiv(m, m1t) * CeilDiv(n, n1);
            if (blocks <= maxBlocks - coreNum) {
                m1 = m1t;
            }
        }
        BalanceWorkload(n, m, n1, m1, 64, coreNum);
        BalanceWorkload(m, n, m1, n1, 64, coreNum); 
    }
    if (k >= 65536) {
        if (m < n || (ratio < 0.1 && n >= 256)) {
            m1 = 128;
            n1 = 256;
        } else {
            m1 = 256;
            n1 = 128;
        }
    }
    k1 = GetMaxK1<fp16_t>(m1, n1);
    L1Shape.m() = m1;
    L1Shape.n() = n1;
    L1Shape.k() = k1;
}

void saveData(std::string filePath, fp16_t *data, size_t fileSize)
{
    std::ofstream outFile(filePath, std::ios::binary);
    if (!outFile) {
        std::cerr << "无法打开文件，文件名：" << filePath << std::endl;
    } else {
        outFile.write(reinterpret_cast<const char*>(data), fileSize);
    }
    outFile.close();
}

static void Run(const Options &options) {
    int smStatus = SHMEM_SUCCESS;
    uint64_t local_mem_size = 2048UL * 1024UL * 1024;

    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.rankId));
    ACL_CHECK(aclrtCreateStream(&stream));

    smStatus = shmem_set_conf_store_tls(false, nullptr, 0);
    shmem_init_attr_t *attributes;
    smStatus = shmem_set_attr(options.rankId % options.rankSize, options.rankSize, local_mem_size, options.ipport, &attributes);
    if (smStatus != SHMEM_SUCCESS) {
        std::cout << "[ERROR] demo run failed!1" << std::endl;
        std::exit(smStatus);
    }
    smStatus = shmem_init_attr(attributes);
    if (smStatus != SHMEM_SUCCESS) {
        std::cout << "[ERROR] demo run failed!2" << std::endl;
        std::exit(smStatus);
    }
    smStatus = shmem_init_status();
    if (smStatus == SHMEM_STATUS_IS_INITIALIZED) {
        std::cout << "[SUCCESS] Shmem init success!" << std::endl;
    } else {
        std::cout << "[ERROR] demo run failed!3" << std::endl;
        std::exit(smStatus);
    }

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    uint32_t rankSize = options.rankSize;
    uint32_t rankId = options.rankId;
    uint32_t repeat_time = options.repeat_time;

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    // aicCoreNum = 1;

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;
    size_t lenF = 3 * 2 * aicCoreNum * rankSize * 512 * 100;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);
    size_t sizeF = lenF * sizeof(int32_t);
    size_t sizeWorkspace;

    const uint32_t align = 256;
    using LayoutTagA = layout::RowMajor;
    // using LayoutTagB = layout::RowMajor;
    using LayoutTagB = layout::ColumnMajor;
    using LayoutTagC = layout::RowMajor;
    LayoutTagA tagA{m, k};
    LayoutTagB tagB{k, n};
    LayoutTagC tagC{m, n};
    bool isNeedPaddingA = false;
    bool isNeedPaddingB = false;
    if constexpr (std::is_same_v<LayoutTagB, layout::RowMajor>){
        isNeedPaddingA = IsNeedPadding(tagA, align);
        isNeedPaddingB = (IsNeedPadding(tagB, align) && (RoundUp(n, (uint32_t)256) * RoundUp(k, (uint32_t)256) * 2 / 48 < 2048 * 1024)
                          && (RoundUp(n, (uint32_t)256) * RoundUp(k, (uint32_t)256) * 2 / 48 > 128 * 1024));
    }

    GemmCoord L1Shape{128, 256, 256};
    if constexpr (std::is_same_v<LayoutTagA, layout::RowMajor> && std::is_same_v<LayoutTagB, layout::ColumnMajor>){
        DoTilingB16Layout01(options.problemShape, L1Shape, aicCoreNum);
    }
    // } else if constexpr (std::is_same_v<LayoutTagA, layout::RowMajor> && std::is_same_v<LayoutTagB, layout::RowMajor>){
    //     DoTilingB16Layout00(options.problemShape, L1Shape, aicCoreNum);
    // }

    size_t sizeWA = GetWorkspaceLen(tagA, L1Shape.m(), L1Shape.k()) * sizeof(fp16_t);
    size_t sizeWB = GetWorkspaceLen(tagB, L1Shape.k(), L1Shape.n()) * sizeof(fp16_t);

    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    std::vector<fp16_t> hostCtemp(lenC);
    golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);
    // golden::FillRandomData<fp16_t>(hostA, 0.1f, 0.1f);
    // golden::FillRandomData<fp16_t>(hostB, 0.1f, 0.1f);
    golden::FillRandomData(hostCtemp, 0.0f, 0.0f);
    std::vector<int> signal(lenF, 0);

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));


    uint8_t *deviceWA{nullptr};
    if (isNeedPaddingA) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST));
    } else {
        // no need to padding A
        deviceWA = deviceA;
    }

    uint8_t *deviceWB{nullptr};
    // If layoutWB has the same stride with layoutB, no need to padding B
    if (isNeedPaddingB) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST));
    } else {
        // no need to padding B
        deviceWB = deviceB;
    }

    void *flagPtr = shmem_malloc(sizeF);
    uint8_t *signalPtr = (uint8_t *)flagPtr;
    ACL_CHECK(aclrtMemcpy(signalPtr, sizeF, signal.data(), sizeF, ACL_MEMCPY_HOST_TO_DEVICE));

    void *symmPtr = shmem_malloc(1024 * 1024 * 1024); // 1024 * 1024 KB 
    uint8_t *symmtricPtr = (uint8_t *)symmPtr;

    uint8_t *deviceWorkspace{nullptr};
    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    // auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ElementA = half;
    using ElementB = half;
    using ElementC = half;
    using ArchTag = Arch::AtlasA2;

    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutB = MakeLayoutFromTag(tagB);
    auto layoutC = MakeLayoutFromTag(tagC);
    using TensorA =
        Tensor<AscendC::GlobalTensor<ElementA>, decltype(layoutA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorB =
        Tensor<AscendC::GlobalTensor<ElementB>, decltype(layoutB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorC =
        Tensor<AscendC::GlobalTensor<ElementC>, decltype(layoutC), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    // using DispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using MmadDispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using CommDispatchPolicy = Distributed::CommAtlasA2AllReduce;

    using BlockComm = Distributed::Block::BlockComm<CommDispatchPolicy, TensorC, TensorC>;

    if (!isNeedPaddingA && !isNeedPaddingB) {
        // no need to padding A and B.
        auto layoutWA = MakeLayout(layoutA.shape(), layoutA.stride());
        auto layoutWB = MakeLayout(layoutB.shape(), layoutB.stride());
        using TensorWA = Tensor<
            AscendC::GlobalTensor<ElementA>, decltype(layoutWA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TensorWB = Tensor<
            AscendC::GlobalTensor<ElementB>, decltype(layoutWB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<
            ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB, TensorC, LayoutTagC, void, void, false, false>;
        // using BlockMmad = Gemm::Block::BlockMmadTla<
        //     DispatchPolicy, L1TileShape, L0TileShape, TensorWA, TensorWB, TensorC, void, TileCopy>;
        using BlockMmad = Gemm::Block::BlockMmadTlaDynamic<MmadDispatchPolicy, TensorWA, TensorWB,
            TensorC, void, TileCopy>;
        using PaddingA = void;
        using PaddingB = void;
        if (options.problemShape.m() > options.problemShape.n()) {
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
            using BlockEpilogue = void;

            // kernel level
            using MatmulKernel = Distributed::Kernel::MatmulAllReduceTla<BlockMmad, BlockEpilogue, BlockScheduler, PaddingA, PaddingB, BlockComm>;

            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

            MatmulKernel::Arguments arguments{options.problemShape, 
                                              L1Shape, 
                                              options.commBlockShape, 
                                              deviceA, 
                                              layoutA, 
                                              deviceB, 
                                              layoutB, 
                                              deviceC, 
                                              layoutC, 
                                              deviceWA, 
                                              layoutWA, 
                                              deviceWB, 
                                              layoutWB, 
                                              symmtricPtr, 
                                              rankSize, 
                                              signalPtr};

            MatmulAdapter matmulOp;
            matmulOp.CanImplement(arguments);
            sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            for (int i = 0 ; i < repeat_time; i++) {
                ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
                ACL_CHECK(aclrtMemcpy(symmtricPtr, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
                // ACL_CHECK(aclrtMemcpy(signalPtr, sizeF, signal.data(), sizeF, ACL_MEMCPY_HOST_TO_DEVICE));
                if (options.enableProf) {
                    ACL_CHECK(aclrtMemset(deviceB, sizeB, 0x1, sizeB));
                }
                matmulOp.Initialize(arguments, deviceWorkspace);
                matmulOp(stream, aicCoreNum, fftsAddr);
                ACL_CHECK(aclrtSynchronizeStream(stream));
            }
        } else {
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
            using BlockEpilogue = void;

            // kernel level
            using MatmulKernel = Distributed::Kernel::MatmulAllReduceTla<BlockMmad, BlockEpilogue, BlockScheduler, PaddingA, PaddingB, BlockComm>;

            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        
            MatmulKernel::Arguments arguments{options.problemShape, 
                                              L1Shape, 
                                              options.commBlockShape, 
                                              deviceA, 
                                              layoutA, 
                                              deviceB, 
                                              layoutB, 
                                              deviceC, 
                                              layoutC, 
                                              deviceWA, 
                                              layoutWA, 
                                              deviceWB, 
                                              layoutWB, 
                                              symmtricPtr, 
                                              rankSize, 
                                              signalPtr};
            MatmulAdapter matmulOp;
            matmulOp.CanImplement(arguments);
            sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            for (int i = 0 ; i < repeat_time; i++) {
                // ACL_CHECK(aclrtSynchronizeStream(stream));
                ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
                ACL_CHECK(aclrtMemcpy(symmtricPtr, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
                // ACL_CHECK(aclrtMemcpy(signalPtr, sizeF, signal.data(), sizeF, ACL_MEMCPY_HOST_TO_DEVICE));
                if (options.enableProf) {
                    ACL_CHECK(aclrtMemset(deviceB, sizeB, 0x1, sizeB));
                }
                matmulOp.Initialize(arguments, deviceWorkspace);
                matmulOp(stream, aicCoreNum, fftsAddr);
                ACL_CHECK(aclrtSynchronizeStream(stream));
            }
        }
    } else if (!isNeedPaddingA && isNeedPaddingB) {
        // no need to padding A, but B needs padding.
        auto layoutWA = MakeLayout(layoutA.shape(), layoutA.stride());
        auto layoutWB = GetPaddingLayout(tagB, L1Shape.k(), L1Shape.n());
        using TensorWA = Tensor<
            AscendC::GlobalTensor<ElementA>, decltype(layoutWA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TensorWB = Tensor<
            AscendC::GlobalTensor<ElementB>, decltype(layoutWB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<
            ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB, TensorC, LayoutTagC, void, void, false, true>;
        // using BlockMmad = Gemm::Block::BlockMmadTla<
        //     DispatchPolicy, L1TileShape, L0TileShape, TensorWA, TensorWB, TensorC, void, TileCopy>;
        using BlockMmad = Gemm::Block::BlockMmadTlaDynamic<MmadDispatchPolicy, TensorWA, TensorWB,
            TensorC, void, TileCopy>;
        using PaddingA = void;
        constexpr const uint32_t computeLengthB = 96 * 1024 / sizeof(ElementB);
        using PaddingB = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorB, TensorWB, computeLengthB>;
        if (options.problemShape.m() > options.problemShape.n()) {
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
            using BlockEpilogue = void;

            // kernel level
            using MatmulKernel = Distributed::Kernel::MatmulAllReduceTla<BlockMmad, BlockEpilogue, BlockScheduler, PaddingA, PaddingB, BlockComm>;

            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        
            MatmulKernel::Arguments arguments{options.problemShape, 
                                              L1Shape, 
                                              options.commBlockShape, 
                                              deviceA, 
                                              layoutA, 
                                              deviceB, 
                                              layoutB, 
                                              deviceC, 
                                              layoutC, 
                                              deviceWA, 
                                              layoutWA, 
                                              deviceWB, 
                                              layoutWB, 
                                              symmtricPtr, 
                                              rankSize, 
                                              signalPtr};
            MatmulAdapter matmulOp;
            matmulOp.CanImplement(arguments);
            sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            for (int i = 0 ; i < repeat_time; i++) {
                // ACL_CHECK(aclrtSynchronizeStream(stream));
                ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
                ACL_CHECK(aclrtMemcpy(symmtricPtr, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
                // ACL_CHECK(aclrtMemcpy(signalPtr, sizeF, signal.data(), sizeF, ACL_MEMCPY_HOST_TO_DEVICE));
                if (options.enableProf) {
                    ACL_CHECK(aclrtMemset(deviceB, sizeB, 0x1, sizeB));
                }
                matmulOp.Initialize(arguments, deviceWorkspace);
                matmulOp(stream, aicCoreNum, fftsAddr);
                ACL_CHECK(aclrtSynchronizeStream(stream));
            }
        } else {
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
            using BlockEpilogue = void;

            // kernel level
            using MatmulKernel = Distributed::Kernel::MatmulAllReduceTla<BlockMmad, BlockEpilogue, BlockScheduler, PaddingA, PaddingB, BlockComm>;

            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        
            MatmulKernel::Arguments arguments{options.problemShape, 
                                              L1Shape, 
                                              options.commBlockShape, 
                                              deviceA, 
                                              layoutA, 
                                              deviceB, 
                                              layoutB, 
                                              deviceC, 
                                              layoutC, 
                                              deviceWA, 
                                              layoutWA, 
                                              deviceWB, 
                                              layoutWB, 
                                              symmtricPtr, 
                                              rankSize, 
                                              signalPtr};
            MatmulAdapter matmulOp;
            matmulOp.CanImplement(arguments);
            sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            for (int i = 0 ; i < repeat_time; i++) {
                // ACL_CHECK(aclrtSynchronizeStream(stream));
                ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
                ACL_CHECK(aclrtMemcpy(symmtricPtr, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
                // ACL_CHECK(aclrtMemcpy(signalPtr, sizeF, signal.data(), sizeF, ACL_MEMCPY_HOST_TO_DEVICE));
                if (options.enableProf) {
                    ACL_CHECK(aclrtMemset(deviceB, sizeB, 0x1, sizeB));
                }
                matmulOp.Initialize(arguments, deviceWorkspace);
                matmulOp(stream, aicCoreNum, fftsAddr);
                ACL_CHECK(aclrtSynchronizeStream(stream));
            }
        }
    } else if (isNeedPaddingA && !isNeedPaddingB) {
        // no need to padding B, but A needs padding.
        auto layoutWA = GetPaddingLayout(tagA, L1Shape.m(), L1Shape.k());
        auto layoutWB = MakeLayout(layoutB.shape(), layoutB.stride());
        using TensorWA = Tensor<
            AscendC::GlobalTensor<ElementA>, decltype(layoutWA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TensorWB = Tensor<
            AscendC::GlobalTensor<ElementB>, decltype(layoutWB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<
            ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB, TensorC, LayoutTagC, void, void, true, false>;
        // using BlockMmad = Gemm::Block::BlockMmadTla<
        //     DispatchPolicy, L1TileShape, L0TileShape, TensorWA, TensorWB, TensorC, void, TileCopy>;
        using BlockMmad = Gemm::Block::BlockMmadTlaDynamic<MmadDispatchPolicy, TensorWA, TensorWB,
            TensorC, void, TileCopy>;
        constexpr const uint32_t computeLengthA = 96 * 1024 / sizeof(ElementA);
        using PaddingA = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorA, TensorWA, computeLengthA>;
        using PaddingB = void;
        if (options.problemShape.m() > options.problemShape.n()) {
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
            using BlockEpilogue = void;

            // kernel level
            using MatmulKernel = Distributed::Kernel::MatmulAllReduceTla<BlockMmad, BlockEpilogue, BlockScheduler, PaddingA, PaddingB, BlockComm>;

            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        
            MatmulKernel::Arguments arguments{options.problemShape, 
                                              L1Shape, 
                                              options.commBlockShape, 
                                              deviceA, 
                                              layoutA, 
                                              deviceB, 
                                              layoutB, 
                                              deviceC, 
                                              layoutC, 
                                              deviceWA, 
                                              layoutWA, 
                                              deviceWB, 
                                              layoutWB, 
                                              symmtricPtr, 
                                              rankSize, 
                                              signalPtr};
            MatmulAdapter matmulOp;
            matmulOp.CanImplement(arguments);
            sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            for (int i = 0 ; i < repeat_time; i++) {
                // ACL_CHECK(aclrtSynchronizeStream(stream));
                ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
                ACL_CHECK(aclrtMemcpy(symmtricPtr, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
                // ACL_CHECK(aclrtMemcpy(signalPtr, sizeF, signal.data(), sizeF, ACL_MEMCPY_HOST_TO_DEVICE));
                if (options.enableProf) {
                    ACL_CHECK(aclrtMemset(deviceB, sizeB, 0x1, sizeB));
                }
                matmulOp.Initialize(arguments, deviceWorkspace);
                matmulOp(stream, aicCoreNum, fftsAddr);
                ACL_CHECK(aclrtSynchronizeStream(stream));
            }
        } else {
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
            using BlockEpilogue = void;

            // kernel level
            using MatmulKernel = Distributed::Kernel::MatmulAllReduceTla<BlockMmad, BlockEpilogue, BlockScheduler, PaddingA, PaddingB, BlockComm>;

            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        
            MatmulKernel::Arguments arguments{options.problemShape, 
                                              L1Shape, 
                                              options.commBlockShape, 
                                              deviceA, 
                                              layoutA, 
                                              deviceB, 
                                              layoutB, 
                                              deviceC, 
                                              layoutC, 
                                              deviceWA, 
                                              layoutWA, 
                                              deviceWB, 
                                              layoutWB, 
                                              symmtricPtr, 
                                              rankSize, 
                                              signalPtr};
            MatmulAdapter matmulOp;
            matmulOp.CanImplement(arguments);
            sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            for (int i = 0 ; i < repeat_time; i++) {
                // ACL_CHECK(aclrtSynchronizeStream(stream));
                ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
                ACL_CHECK(aclrtMemcpy(symmtricPtr, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
                // ACL_CHECK(aclrtMemcpy(signalPtr, sizeF, signal.data(), sizeF, ACL_MEMCPY_HOST_TO_DEVICE));
                if (options.enableProf) {
                    ACL_CHECK(aclrtMemset(deviceB, sizeB, 0x1, sizeB));
                }
                matmulOp.Initialize(arguments, deviceWorkspace);
                matmulOp(stream, aicCoreNum, fftsAddr);
                ACL_CHECK(aclrtSynchronizeStream(stream));
            }
        }
    } else {
        // Both A and B need padding.
        auto layoutWA = GetPaddingLayout(tagA, L1Shape.m(), L1Shape.k());
        auto layoutWB = GetPaddingLayout(tagB, L1Shape.k(), L1Shape.n());
        using TensorWA = Tensor<
            AscendC::GlobalTensor<ElementA>, decltype(layoutWA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TensorWB = Tensor<
            AscendC::GlobalTensor<ElementB>, decltype(layoutWB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<
            ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB, TensorC, LayoutTagC, void, void, true, true>;
        // using BlockMmad = Gemm::Block::BlockMmadTla<
        //     DispatchPolicy, L1TileShape, L0TileShape, TensorWA, TensorWB, TensorC, void, TileCopy>;
        using BlockMmad = Gemm::Block::BlockMmadTlaDynamic<MmadDispatchPolicy, TensorWA, TensorWB,
            TensorC, void, TileCopy>;
        constexpr const uint32_t computeLengthA = 96 * 1024 / sizeof(ElementA);
        using PaddingA = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorA, TensorWA, computeLengthA>;
        constexpr const uint32_t computeLengthB = 96 * 1024 / sizeof(ElementB);
        using PaddingB = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorB, TensorWB, computeLengthB>;
        if (options.problemShape.m() > options.problemShape.n()) {
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
            using BlockEpilogue = void;

            // kernel level
            using MatmulKernel = Distributed::Kernel::MatmulAllReduceTla<BlockMmad, BlockEpilogue, BlockScheduler, PaddingA, PaddingB, BlockComm>;

            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        
            MatmulKernel::Arguments arguments{options.problemShape, 
                                              L1Shape, 
                                              options.commBlockShape, 
                                              deviceA, 
                                              layoutA, 
                                              deviceB, 
                                              layoutB, 
                                              deviceC, 
                                              layoutC, 
                                              deviceWA, 
                                              layoutWA, 
                                              deviceWB, 
                                              layoutWB, 
                                              symmtricPtr, 
                                              rankSize, 
                                              signalPtr};
            MatmulAdapter matmulOp;
            matmulOp.CanImplement(arguments);
            sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            for (int i = 0 ; i < repeat_time; i++) {
                // ACL_CHECK(aclrtSynchronizeStream(stream));
                ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
                ACL_CHECK(aclrtMemcpy(symmtricPtr, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
                // ACL_CHECK(aclrtMemcpy(signalPtr, sizeF, signal.data(), sizeF, ACL_MEMCPY_HOST_TO_DEVICE));
                if (options.enableProf) {
                    ACL_CHECK(aclrtMemset(deviceB, sizeB, 0x1, sizeB));
                }
                matmulOp.Initialize(arguments, deviceWorkspace);
                matmulOp(stream, aicCoreNum, fftsAddr);
                ACL_CHECK(aclrtSynchronizeStream(stream));
            }
        } else {
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
            using BlockEpilogue = void;

            // kernel level
            using MatmulKernel = Distributed::Kernel::MatmulAllReduceTla<BlockMmad, BlockEpilogue, BlockScheduler, PaddingA, PaddingB, BlockComm>;

            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        
            MatmulKernel::Arguments arguments{options.problemShape, 
                                              L1Shape, 
                                              options.commBlockShape, 
                                              deviceA, 
                                              layoutA, 
                                              deviceB, 
                                              layoutB, 
                                              deviceC, 
                                              layoutC, 
                                              deviceWA, 
                                              layoutWA, 
                                              deviceWB, 
                                              layoutWB, 
                                              symmtricPtr, 
                                              rankSize, 
                                              signalPtr};
            MatmulAdapter matmulOp;
            matmulOp.CanImplement(arguments);
            sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            for (int i = 0 ; i < repeat_time; i++) {
                // ACL_CHECK(aclrtSynchronizeStream(stream));
                ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
                ACL_CHECK(aclrtMemcpy(symmtricPtr, sizeC, hostCtemp.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
                // ACL_CHECK(aclrtMemcpy(signalPtr, sizeF, signal.data(), sizeF, ACL_MEMCPY_HOST_TO_DEVICE));
                if (options.enableProf) {
                    ACL_CHECK(aclrtMemset(deviceB, sizeB, 0x1, sizeB));
                }
                matmulOp.Initialize(arguments, deviceWorkspace);
                matmulOp(stream, aicCoreNum, fftsAddr);
                ACL_CHECK(aclrtSynchronizeStream(stream));
            }
        }
    }
    // ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    // ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, symmtricPtr, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::string dataPath = "./examples/41_matmul_allreduce/data/";
    std::string aFilePath = dataPath + "a" + std::to_string(options.rankId) + ".bin";
    std::string bFilePath = dataPath + "b" + std::to_string(options.rankId) + ".bin";
    std::string resultFilePath = dataPath + "result" + std::to_string(options.rankId) + ".bin";
    saveData(aFilePath, hostA.data(), sizeA);
    saveData(bFilePath, hostB.data(), sizeB);
    saveData(resultFilePath, hostC.data(), sizeC);

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    if (isNeedPaddingA) {
        ACL_CHECK(aclrtFree(deviceWA));
    }
    if (isNeedPaddingB) {
        ACL_CHECK(aclrtFree(deviceWB));
    }

    shmem_free(symmPtr);
    shmem_free(flagPtr);
    smStatus = shmem_finalize();

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.rankId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank_;
    int rankSize_;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &rankSize_);

    Options options;
    if (options.Parse(argc, argv, rank_, rankSize_) != 0) {
        return -1;
    }
    
    Run(options);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
