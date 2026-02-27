#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <sstream>
#include <fstream>

#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "catlass/distributed/kernel/grouped_matmul_alltoallv.hpp"
#include "catlass/distributed/dispatch_policy.hpp"
#include "catlass/distributed/block/block_comm.hpp"

#include "shmem_api.h"

#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_rma.h"
#include "host/shmem_host_team.h"

using namespace Catlass;
using namespace tla;
using fp16_t = op::fp16_t;

std::ostringstream oss;

void saveData(std::string filePath, fp16_t *data, size_t fileSize)
{
    std::ofstream outFile(filePath, std::ios::binary);
    if (!outFile)
    {
        std::cerr << "无法打开文件，文件名：" << filePath << std::endl;
    }
    else
    {
        outFile.write(reinterpret_cast<const char *>(data), fileSize);
    }
    outFile.close();
}

void saveData(std::string filePath, uint32_t *data, size_t fileSize)
{
    std::ofstream outFile(filePath, std::ios::binary);
    if (!outFile)
    {
        std::cerr << "无法打开文件，文件名：" << filePath << std::endl;
    }
    else
    {
        outFile.write(reinterpret_cast<const char *>(data), fileSize);
    }
    outFile.close();
}

struct Options
{
    const std::string HELPER = "27_grouped_matmul_alltoallv m n k groupsize rankSize";

    GemmCoord problemShape{128, 128, 128};
    uint32_t groupSize;
    uint32_t rankSize;
    uint32_t rankId;
    const char *ipport;
    uint32_t repeat_time;

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex
        {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            GROUPSIZE_INDEX,
            RANKSIZE_INDEX,
            RANKID_INDEX,
            IPPORT_INDEX,
            REPEAT_NUM,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= RANKID_INDEX)
        {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        groupSize = std::atoi(argv[GROUPSIZE_INDEX]);
        rankSize = std::atoi(argv[RANKSIZE_INDEX]);
        rankId = std::atoi(argv[RANKID_INDEX]);
        ipport = argv[IPPORT_INDEX];
        repeat_time = std::atoi(argv[REPEAT_NUM]);

        std::ostringstream oss;
        oss << "argv: " << problemShape.m() << " " << problemShape.n() << " " << problemShape.k() << " "
            << rankSize << " " << rankId << " " << ipport << " " << "\n";

        return 0;
    }
};

void Run(Options const &options)
{
    int smStatus = SHMEM_SUCCESS;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024; // shmem空间大小

    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.rankId));
    ACL_CHECK(aclrtCreateStream(&stream));

    shmem_init_attr_t *attributes;
    shmem_set_attr(options.rankId % options.rankSize, options.rankSize, local_mem_size, options.ipport, &attributes);
    if (smStatus != SHMEM_SUCCESS)
    {
        std::cout << "[ERROR] set_attr failed!" << std::endl;
        std::exit(smStatus);
    }
    shmem_init_attr(attributes);
    if (smStatus != SHMEM_SUCCESS)
    {
        std::cout << "[ERROR] init_attr failed!" << std::endl;
        std::exit(smStatus);
    }
    smStatus = shmem_init_status();
    if (smStatus == SHMEM_STATUS_IS_INITALIZED)
    {
        std::cout << "[SUCCESS] Shmem init success!" << std::endl;
    }
    else
    {
        std::cout << "[ERROR] shmem init failed!" << std::endl;
        std::exit(smStatus);
    }

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    uint32_t rankSize = options.rankSize;
    uint32_t rankId = options.rankId;
    uint32_t repeat_time = options.repeat_time;

    uint32_t localExpertNum = options.groupSize;  // 每张卡上的专家数量
    uint32_t epSize = rankSize;                   // 专家并行度即rank数
    uint32_t expertNum = localExpertNum * epSize; // 全局专家数量
    uint32_t m_ = m * rankSize;                   // alltoallv前,一张卡张最多持有的token数量

    // 生成一个(ranksize， 全局专家数量)大小的表，名为globalTokensPerExpertData。 每个元素代该专家要发给该rank的token数量。
    std::vector<uint32_t> globalTokensPerExpertData;
    globalTokensPerExpertData.reserve(epSize * expertNum);
    int seed = 0;
    std::mt19937 gen(seed); // 准备随机数种子，需要保证所有卡一致，以确保生成的 global tokens 一致
    std::uniform_int_distribution<uint32_t> distribution(0, m);
    std::vector<uint32_t> tokensAccum(expertNum + 1);
    tokensAccum[0] = 0;
    tokensAccum[expertNum] = m;
    for (int epIdx = 0; epIdx < epSize; ++epIdx)
    {
        for (int i = 1; i < expertNum; ++i)
        {
            tokensAccum[i] = distribution(gen);
        }
        std::sort(tokensAccum.begin() + 1, tokensAccum.begin() + expertNum);
        for (int i = 0; i < expertNum; ++i)
        {
            globalTokensPerExpertData.push_back(tokensAccum[i + 1] - tokensAccum[i]);
        }
    }

    // rank0打印globalTokens表
    if (rankId == 0)
    {
        oss << "Global tokens per expert:\n";
        for (int epIdx = 0; epIdx < epSize; ++epIdx)
        {
            int epExpertStart = epIdx * expertNum;
            oss << globalTokensPerExpertData[epExpertStart];
            for (int expertIdx = 1; expertIdx < expertNum; ++expertIdx)
            {
                oss << " " << globalTokensPerExpertData[epExpertStart + expertIdx];
            }
            oss << "\n";
        }
    }

    size_t lenA = static_cast<size_t>(m_) * k;
    size_t lenB = static_cast<size_t>(k) * n * localExpertNum;
    size_t lenWC = static_cast<size_t>(m_) * n;
    size_t lenC = static_cast<size_t>(m) * n;
    size_t lenglobalTokensPerExpertData = epSize * expertNum;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeWC = lenWC * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);
    size_t sizeGlobalTokensPerExpert = lenglobalTokensPerExpertData * sizeof(uint32_t);
    size_t sizeWorkspace;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    // using LayoutTagB = layout::ColumnMajor;
    using LayoutTagWC = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;

    LayoutTagA tagA{m_, k};
    LayoutTagB tagB{k, n};
    LayoutTagWC tagWC{m_, n};
    LayoutTagC tagC{m, n};

    // 填入随机数据
    std::vector<fp16_t> hostA(lenA, 0.0);
    std::vector<fp16_t> hostB(lenB, 1.0);
    golden::FillRandomData(hostA, -1.0f, 1.0f);
    golden::FillRandomData(hostB, -1.0f, 1.0f);

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceWC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWC), sizeWC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *globalTokensPerExpert{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&globalTokensPerExpert), sizeGlobalTokensPerExpert, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(globalTokensPerExpert, sizeGlobalTokensPerExpert, globalTokensPerExpertData.data(), sizeGlobalTokensPerExpert, ACL_MEMCPY_HOST_TO_DEVICE));

    void *symmPtr = shmem_malloc(1024 * 1024 * 1024); // 1024 * 1024 KB
    uint8_t *symmtricPtr = (uint8_t *)symmPtr;

    uint8_t *deviceWorkspace{nullptr};

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;

    using L1TileShape = Shape<_128, _256, _256>;
    using L0TileShape = Shape<_128, _256, _64>;

    using ElementA = half;
    using ElementB = half;
    using ElementWC = half;
    using ElementC = half;

    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutB = MakeLayoutFromTag(tagB);
    auto layoutWC = MakeLayoutFromTag(tagWC);
    auto layoutC = MakeLayoutFromTag(tagC);

    using TensorA = Tensor<AscendC::GlobalTensor<ElementA>, decltype(layoutA), AscendC::TPosition::GM>;
    using TensorB = Tensor<AscendC::GlobalTensor<ElementB>, decltype(layoutB), AscendC::TPosition::GM>;
    using TensorWC = Tensor<AscendC::GlobalTensor<ElementWC>, decltype(layoutWC), AscendC::TPosition::GM>;
    using TensorC = Tensor<AscendC::GlobalTensor<ElementC>, decltype(layoutC), AscendC::TPosition::GM>;

    using TileCopy =
        Gemm::Tile::PackedTileCopyTla<ArchTag, TensorA, LayoutTagA, TensorB, LayoutTagB, TensorWC, LayoutTagWC>; // matmul时，搬AB块，结果块放到WC上
    using BlockMmad =
        Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape,
                                  TensorA, TensorB, TensorWC, void, TileCopy>;

    using CommDispatchPolicy = Distributed::CommAtlasA2All2Allv;
    using BlockComm =
        Distributed::Block::BlockComm<CommDispatchPolicy, TensorWC, TensorC>;

    using BlockEpilogue = void;

    if (options.problemShape.m() > options.problemShape.n())
    {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        // kernel level
        using MatmulKernel = Distributed::Kernel::GroupedMatmulAlltoAllvTla<BlockMmad, BlockEpilogue, BlockScheduler, BlockComm>; // TODO
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulKernel::Arguments arguments{
            options.problemShape, localExpertNum, globalTokensPerExpert,
            deviceA, layoutA, deviceB, layoutB, deviceWC, layoutWC, symmtricPtr, layoutC, rankSize, deviceC};
        MatmulAdapter matmul_op;
        matmul_op.CanImplement(arguments);
        uint64_t sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0)
        {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        matmul_op.Initialize(arguments, deviceWorkspace);
        for (int i = 0; i < repeat_time; i++)
        {
            matmul_op(stream, aicCoreNum, fftsAddr);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
    }
    else
    {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        // kernel level
        using MatmulKernel = Distributed::Kernel::GroupedMatmulAlltoAllvTla<BlockMmad, BlockEpilogue, BlockScheduler, BlockComm>; // TODO
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulKernel::Arguments arguments{
            options.problemShape, localExpertNum, globalTokensPerExpert,
            deviceA, layoutA, deviceB, layoutB, deviceWC, layoutWC, symmtricPtr, layoutC, rankSize, deviceC};
        MatmulAdapter matmul_op;
        matmul_op.CanImplement(arguments);
        sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0)
        {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        matmul_op.Initialize(arguments, deviceWorkspace);
        for (int i = 0; i < repeat_time; i++)
        {
            matmul_op(stream, aicCoreNum, fftsAddr);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
    }

    // 搬回通信结果
    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    // 验证groupedmatmul结果
    std::vector<fp16_t> hostWC(lenWC);
    ACL_CHECK(aclrtMemcpy(hostWC.data(), sizeWC, deviceWC, sizeWC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::cout << oss.str();

    std::string dataPath = "./examples/27_grouped_matmul_alltoallv/data/";
    std::string aFilePath = dataPath + "a" + std::to_string(options.rankId) + ".bin";
    std::string bFilePath = dataPath + "b" + std::to_string(options.rankId) + ".bin";
    std::string wcFilePath = dataPath + "wc" + std::to_string(options.rankId) + ".bin";
    std::string resultFilePath = dataPath + "result" + std::to_string(options.rankId) + ".bin";
    std::string globalTokensPerExpertFilePath = dataPath + "globalTokensPerExpert" + ".bin";
    saveData(aFilePath, hostA.data(), sizeA);
    saveData(bFilePath, hostB.data(), sizeB);
    saveData(wcFilePath, hostWC.data(), sizeWC);
    saveData(resultFilePath, hostC.data(), sizeC);
    if (rankId == 0)
    {
        saveData(globalTokensPerExpertFilePath, globalTokensPerExpertData.data(), sizeGlobalTokensPerExpert);
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceWC));

    ACL_CHECK(aclrtFree(globalTokensPerExpert));

    shmem_free(symmPtr);
    shmem_finalize();

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(rankId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0)
    {
        std::cout << "Option is not valid!\n";
        return -1;
    }
    Run(options);

    std::cout << "GmmAlltoAllv Demo run success!!\n";
    return 0;
}