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
#include "catlass/distributed/kernel/allgather_matmul.hpp"
#include "catlass/layout/layout.hpp"

#include "helper.hpp"
#include "golden.hpp"

#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_rma.h"
#include "host/shmem_host_team.h"

#include "shmem_api.h"

using namespace Catlass;
using namespace tla;

template<class Layout>
auto GetPaddingLayout(Layout layout, uint32_t blockRows, uint32_t blockCols)
{
    if constexpr (std::is_same_v<Layout, layout::RowMajor>) {
        auto shape = MakeShape(MakeShape(blockRows, CeilDiv(layout.shape(0), blockRows)),
            MakeShape(blockCols, CeilDiv(layout.shape(1), blockCols)));
        auto stride = MakeStride(
            MakeStride(
                static_cast<int64_t>(blockCols),
                static_cast<int64_t>(blockRows) * RoundUp(layout.shape(1), blockCols)
            ),
            MakeStride(Int<1>{}, static_cast<int64_t>(blockRows) * blockCols)
        );
        return MakeLayout(shape, stride);
    } else {
        auto shape = MakeShape(MakeShape(blockRows, CeilDiv(layout.shape(0), blockRows)),
            MakeShape(blockCols, CeilDiv(layout.shape(1), blockCols)));
        auto stride = MakeStride(
            MakeStride(Int<1>{}, static_cast<int64_t>(blockRows) * blockCols),
            MakeStride(
                static_cast<int64_t>(blockRows),
                RoundUp(layout.shape(0), blockRows) * static_cast<int64_t>(blockCols)
            )
        );
        return MakeLayout(shape, stride);
    }
}

using Options = CommAGOptions;

template<class Layout>
auto GetPaddingALayout(Layout layout, uint32_t blockRows, uint32_t blockCols)
{
    if constexpr (std::is_same_v<Layout, layout::RowMajor>) {
        auto shape = MakeShape(layout.shape(0), layout.shape(1));
        auto stride = MakeStride(static_cast<int64_t>(RoundUp((uint32_t)layout.stride(0), blockCols)), Int<1>{});
        // auto stride = MakeStride(layout.stride(0), Int<1>{});
        return MakeLayout(shape, stride);
    } else {
        auto shape = MakeShape(layout.shape(0), layout.shape(1));
        auto stride = MakeStride(Int<1>{}, static_cast<int64_t>(RoundUp((uint32_t)layout.stride(1), blockRows))); 
        // auto stride = MakeStride(Int<1>{}, layout.stride(1));
        return MakeLayout(shape, stride);
    }
}

template<class Layout>
size_t GetWorkspaceLen(Layout layout, size_t blockRows, size_t blockCols)
{
    return RoundUp(static_cast<size_t>(layout.shape(0)), blockRows) *
        RoundUp(static_cast<size_t>(layout.shape(1)), blockCols);
}

template<class Layout>
size_t GetSymmWorkspaceLen(Layout layout, size_t blockCols)
{
    if (std::is_same_v<Layout, layout::RowMajor>) {
        return static_cast<size_t>(layout.shape(0)) * 
            RoundUp(static_cast<size_t>(layout.shape(1)), blockCols);
    } 
    return RoundUp(static_cast<size_t>(layout.shape(0)), blockCols) *
        static_cast<size_t>(layout.shape(1));
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

void Run(Options const &options)
{
    int smStatus = SHMEM_SUCCESS;
    uint64_t local_mem_size = 2048UL * 1024UL * 1024;        // shmem空间大小
    aclrtStream stream{nullptr};
    
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.rankId));
    ACL_CHECK(aclrtCreateStream(&stream));

    smStatus = shmem_set_conf_store_tls(false, nullptr, 0);
    shmem_init_attr_t *attributes;
    smStatus = shmem_set_attr(options.rankId % options.rankSize, options.rankSize, local_mem_size, options.ipport, &attributes);
    if (smStatus != SHMEM_SUCCESS) {
        std::cout << "[ERROR] shmem_set_attr failed!" << std::endl;
        std::exit(smStatus);
    }
    smStatus = shmem_init_attr(attributes);
    if (smStatus != SHMEM_SUCCESS) {
        std::cout << "[ERROR] shmem_init_attr failed!" << std::endl;
        std::exit(smStatus);
    }
    smStatus = shmem_init_status();
    if (smStatus == SHMEM_STATUS_IS_INITIALIZED) {
        std::cout << "[SUCCESS] Shmem init success!" << std::endl;
    } else {
        std::cout << "[ERROR] shmem_init_status failed!" << std::endl;
        std::exit(smStatus);
    }

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    uint32_t rankSize = options.rankSize;
    uint32_t rankId = options.rankId;
    uint32_t repeat_time = options.repeat_time;

    const uint32_t align = 256;
    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n * rankSize;
    size_t lenF = 1024 * 1024;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);
    size_t sizeF = lenF * sizeof(int32_t);
    size_t sizeWorkspace;

    // LayoutTagA和LayoutTagWA保持一致
    using LayoutTagA = layout::RowMajor;
    // using LayoutTagB = layout::RowMajor;
    using LayoutTagB = layout::ColumnMajor;
    using LayoutTagC = layout::RowMajor;
    using LayoutTagSymmetric = layout::RowMajor;
    LayoutTagA tagA{m, k};
    LayoutTagB tagB{k, n};
    LayoutTagC tagC{m * rankSize, n};
    // LayoutTagWA tagWA{m, k};
    LayoutTagSymmetric tagSymmetric{m * rankSize, k};
    bool isNeedPaddingB = IsNeedPadding(tagB, align);

    using L1TileShape = std::conditional_t<std::is_same_v<LayoutTagA, layout::ColumnMajor> && 
        std::is_same_v<LayoutTagB, layout::ColumnMajor>, Shape<_256, _128, _256>, Shape<_128, _256, _256>>;
    size_t sizeSymmetric = GetSymmWorkspaceLen(tagSymmetric, get<2>(L1TileShape{})) * sizeof(fp16_t);
    size_t sizeWB = GetWorkspaceLen(tagB, get<2>(L1TileShape{}), get<1>(L1TileShape{})) * sizeof(fp16_t);

    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    golden::FillRandomData(hostA, -5.0f, 5.0f);
    golden::FillRandomData(hostB, -5.0f, 5.0f);
    std::vector<int32_t> signal(lenF, 0);

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    
    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    void *flagPtr = shmem_malloc(sizeF);
    uint8_t *signalPtr = (uint8_t *)flagPtr;
    ACL_CHECK(aclrtMemcpy(signalPtr, sizeF, signal.data(), sizeF, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceWB{nullptr};
    // If layoutWB has the same stride with layoutB, no need to padding B
    if (isNeedPaddingB) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST));
    } else {
        // no need to padding B
        deviceWB = deviceB;
    }

    void *symmPtr = shmem_malloc(1024 * 1024 * 1024); // 1024 * 1024 KB 
    uint8_t *symmtricPtr = (uint8_t *)symmPtr;
    
    uint8_t *deviceWorkspace{nullptr};

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::AtlasA2;
    using L1TileShape = std::conditional_t<std::is_same_v<LayoutTagA, layout::ColumnMajor> &&
        std::is_same_v<LayoutTagB, layout::ColumnMajor>, Shape<_256, _128, _256>, Shape<_128, _256, _256>>;
    using L0TileShape = std::conditional_t<std::is_same_v<LayoutTagA, layout::ColumnMajor> &&
        std::is_same_v<LayoutTagB, layout::ColumnMajor>, Shape<_256, _128, _64>, Shape<_128, _256, _64>>;

    using ElementA = half;
    using ElementB = half;
    using ElementC = half;

    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutB = MakeLayoutFromTag(tagB);
    auto layoutC = MakeLayoutFromTag(tagC);

    using TensorA = Tensor<AscendC::GlobalTensor<ElementA>, decltype(layoutA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorB = Tensor<AscendC::GlobalTensor<ElementB>, decltype(layoutB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorC = Tensor<AscendC::GlobalTensor<ElementC>, decltype(layoutC), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    // using TensorWA = Tensor<AscendC::GlobalTensor<ElementA>, decltype(layoutWA), AscendC::TPosition::GM>;

    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;

    using MmadDispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using CommDispatchPolicy = Distributed::CommAtlasA2AllGather;

    if (!isNeedPaddingB) {
        auto layoutWA = GetPaddingALayout(tagA, get<0>(L1TileShape{}), get<2>(L1TileShape{}));
        // auto layoutWA = MakeLayoutFromTag(tagSymmetric);
        using TensorWA = Tensor<AscendC::GlobalTensor<ElementA>, decltype(layoutWA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        auto layoutWB = MakeLayout(layoutB.shape(), layoutB.stride());
        using TensorWB = Tensor<AscendC::GlobalTensor<ElementB>, decltype(layoutWB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB,
            TensorC, LayoutTagC, void, void, false, false>;
        using BlockMmad = Gemm::Block::BlockMmadTla<MmadDispatchPolicy, L1TileShape, L0TileShape, TensorWA, TensorWB,
            TensorC, void, TileCopy>;
        using BlockComm = Distributed::Block::BlockComm<CommDispatchPolicy, TensorA, TensorWA>;
        using PaddingB = void;
        if (options.problemShape.m() > options.problemShape.n()) {
            // Swizzle offset is 3 and direction is 0.
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
            using BlockEpilogue = void;

            // kernel level
            using MatmulKernel = Distributed::Kernel::AllGatherMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler, PaddingB, BlockComm>;

            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        
            MatmulKernel::Arguments arguments{
                options.problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC, 
                deviceWB, layoutWB, symmtricPtr, layoutWA, signalPtr
            };

            MatmulAdapter matmul_op;
            matmul_op.CanImplement(arguments);
            sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            matmul_op.Initialize(arguments, deviceWorkspace);
            for (int i = 0; i < repeat_time; i++) {
                if (options.enableProf) {
                    ACL_CHECK(aclrtMemset(deviceB, sizeB, 0x1, sizeB));
                }
                matmul_op(stream, aicCoreNum, fftsAddr);
                ACL_CHECK(aclrtSynchronizeStream(stream));
            }
        } else {
            // Swizzle offset is 3 and direction is 1.
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
            using BlockEpilogue = void;

            // kernel level
            using MatmulKernel = Distributed::Kernel::AllGatherMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler, PaddingB, BlockComm>;

            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        
            MatmulKernel::Arguments arguments{
                options.problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC, 
                deviceWB, layoutWB, symmtricPtr, layoutWA, signalPtr
            };

            MatmulAdapter matmul_op;
            matmul_op.CanImplement(arguments);
            sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            matmul_op.Initialize(arguments, deviceWorkspace);
            for (int i = 0; i < repeat_time; i++) {
                if (options.enableProf) {
                    ACL_CHECK(aclrtMemset(deviceB, sizeB, 0x1, sizeB));
                }
                matmul_op(stream, aicCoreNum, fftsAddr);
                ACL_CHECK(aclrtSynchronizeStream(stream));
            }
        }
    } else if (isNeedPaddingB) {
        auto layoutWA = GetPaddingALayout(tagA, get<0>(L1TileShape{}), get<2>(L1TileShape{}));
        // auto layoutWA = MakeLayoutFromTag(tagSymmetric);
        using TensorWA = Tensor<AscendC::GlobalTensor<ElementA>, decltype(layoutWA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        auto layoutWB = GetPaddingLayout(tagB, get<2>(L1TileShape{}), get<1>(L1TileShape{}));
        using TensorWB = Tensor<AscendC::GlobalTensor<ElementB>, decltype(layoutWB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB,
            TensorC, LayoutTagC, void, void, false, true>;
        using BlockMmad = Gemm::Block::BlockMmadTla<MmadDispatchPolicy, L1TileShape, L0TileShape, TensorWA, TensorWB,
            TensorC, void, TileCopy>;
        using BlockComm = Distributed::Block::BlockComm<CommDispatchPolicy, TensorA, TensorWA>;
        constexpr const uint32_t computeLengthB = 96 * 1024 / sizeof(ElementB);
        using PaddingB = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorB, TensorWB, computeLengthB>;
        if (options.problemShape.m() > options.problemShape.n()) {
            // Swizzle offset is 3 and direction is 0.
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
            using BlockEpilogue = void;

            // kernel level
            using MatmulKernel = Distributed::Kernel::AllGatherMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler, PaddingB, BlockComm>;

            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        
            MatmulKernel::Arguments arguments{
                options.problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC, 
                deviceWB, layoutWB, symmtricPtr, layoutWA, signalPtr
            };

            MatmulAdapter matmul_op;
            matmul_op.CanImplement(arguments);
            sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            matmul_op.Initialize(arguments, deviceWorkspace);
            for (int i = 0; i < repeat_time; i++) {
                if (options.enableProf) {
                    ACL_CHECK(aclrtMemset(deviceB, sizeB, 0x1, sizeB));
                }
                matmul_op(stream, aicCoreNum, fftsAddr);
                ACL_CHECK(aclrtSynchronizeStream(stream));
            }
        } else {
            // Swizzle offset is 3 and direction is 1.
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
            using BlockEpilogue = void;

            // kernel level
            using MatmulKernel = Distributed::Kernel::AllGatherMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler, PaddingB, BlockComm>;

            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        
            MatmulKernel::Arguments arguments{
                options.problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC, 
                deviceWB, layoutWB, symmtricPtr, layoutWA, signalPtr
            };

            MatmulAdapter matmul_op;
            matmul_op.CanImplement(arguments);
            sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
            if (sizeWorkspace > 0) {
                ACL_CHECK(
                    aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
                );
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            matmul_op.Initialize(arguments, deviceWorkspace);
            for (int i = 0; i < repeat_time; i++) {
                if (options.enableProf) {
                    ACL_CHECK(aclrtMemset(deviceB, sizeB, 0x1, sizeB));
                }
                matmul_op(stream, aicCoreNum, fftsAddr);
                ACL_CHECK(aclrtSynchronizeStream(stream));
            }
        }
    }
    // ACL_CHECK(aclrtSynchronizeStream(stream));

    // 从设备端拷贝结果
    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<fp16_t> hostWA(sizeSymmetric / sizeof(fp16_t));
    ACL_CHECK(aclrtMemcpy(hostWA.data(), sizeSymmetric, symmtricPtr, sizeSymmetric, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<int32_t> flag(lenF);
    ACL_CHECK(aclrtMemcpy(flag.data(), sizeF, signalPtr, sizeF, ACL_MEMCPY_DEVICE_TO_HOST));

    // std::cout << "rankId: " << options.rankId << "  flag 0: " << flag[0] << std::endl;
    // for (int i = 1; i < 4; i++) {
    //     std::cout << "rankId: " << options.rankId << "  flag " << i << ": " << flag[i * 32] << std::endl;
    // }

    // 保存数据
    std::string dataPath = "./examples/40_allgather_matmul/data/";
    std::string aFilePath = dataPath + "a" + std::to_string(options.rankId) + ".bin";
    std::string bFilePath = dataPath + "b" + std::to_string(options.rankId) + ".bin";
    std::string AGFilePath = dataPath + "allgather" + std::to_string(options.rankId) + ".bin";
    std::string resultFilePath = dataPath + "result" + std::to_string(options.rankId) + ".bin";
    saveData(aFilePath, hostA.data(), sizeA);
    saveData(AGFilePath, hostWA.data(), sizeSymmetric);
    saveData(bFilePath, hostB.data(), sizeB);
    saveData(resultFilePath, hostC.data(), sizeC);

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    if (isNeedPaddingB) {
        ACL_CHECK(aclrtFree(deviceWB));
    }
    shmem_free(symmPtr);
    shmem_free(flagPtr);
    smStatus = shmem_finalize();

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.rankId));
    ACL_CHECK(aclFinalize());

    std::cout << "[SUCCESS] Rank " << options.rankId << " kernel run success! Data saved in " << dataPath << std::endl;
}

int main(int argc, char **argv)
{
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