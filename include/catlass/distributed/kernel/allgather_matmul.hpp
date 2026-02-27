#ifndef CATLASS_DISTRIBUTE_KERNEL_ALLGATHER_MATMUL_HPP
#define CATLASS_DISTRIBUTE_KERNEL_ALLGATHER_MATMUL_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/kernel/optimized_matmul_tla.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "shmem_api.h"

namespace Catlass::Distributed::Kernel {

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class PaddingB,
    class BlockComm_
>
class AllGatherMatmulTla {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;

    using ElementA = typename BlockMmad::ElementA;      
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;

    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutWA = typename BlockMmad::LayoutA;
    using LayoutWB = typename BlockMmad::LayoutB;

    using BlockComm = BlockComm_;
    using BlockScheduler = BlockScheduler_;
    template<class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template<>
    struct LayoutHelper<void> {
        using type = void;
    };
    using LayoutA = typename LayoutHelper<BlockComm>::type;
    using LayoutB = std::conditional_t<std::is_void_v<PaddingB>, LayoutWB, typename LayoutHelper<PaddingB>::type>;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWB;
        LayoutWB layoutWB;
        GM_ADDR ptrSymmetric;     
        LayoutWA layoutWA;
        // 标志位地址
        GM_ADDR ptrSignal; 

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, 
            GM_ADDR ptrA_, LayoutA layoutA_, 
            GM_ADDR ptrB_, LayoutB layoutB_, 
            GM_ADDR ptrC_, LayoutC layoutC_, 
            GM_ADDR ptrWB_, LayoutWB layoutWB_,
            GM_ADDR ptrSymmetric_, LayoutWA layoutWA_, 
            GM_ADDR ptrSignal_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_), ptrWB(ptrWB_), layoutWB(layoutWB_), ptrSymmetric(ptrSymmetric_), layoutWA(layoutWA_), 
              ptrSignal(ptrSignal_){}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint8_t *ptrA; LayoutA layoutA;
        uint8_t *ptrB; LayoutB layoutB;
        uint8_t *ptrC; LayoutC layoutC;
        uint8_t *ptrWB; LayoutWB layoutWB;
        uint8_t *ptrSymmetric; LayoutWA layoutWA;
        uint8_t *ptrSignal; 
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Params params{args.problemShape,
            args.ptrA, args.layoutA,
            args.ptrB, args.layoutB,
            args.ptrC, args.layoutC,
            args.ptrWB, args.layoutWB,
            args.ptrSymmetric, args.layoutWA,
            args.ptrSignal
        };
        return params;
    }

    CATLASS_DEVICE
    AllGatherMatmulTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    // matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrSymmetric);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        auto tensorA = tla::MakeTensor(gmA, params.layoutWA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutWB, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

        uint32_t rankId = shmem_my_pe();
        uint32_t rankSize = shmem_n_pes();

        Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishAllGather);

        GemmCoord actualCocShape{params.problemShape.m() * rankSize, params.problemShape.n(), params.problemShape.k()};

        BlockScheduler matmulBlockScheduler(actualCocShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // 全同步
        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            auto tensorBlockA = GetTile(
                tensorA,
                tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k())
            );
            auto tensorBlockB = GetTile(
                tensorB,
                tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n())
            );
            auto tensorBlockC = GetTile(
                tensorC,
                tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n())
            );

            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            bool hasNextBlock = false;
            GemmCoord nextBlockCoord;
            GemmCoord nextActualBlockShape;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                nextBlockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockCoord);
            }

            auto nextTensorBlockA = GetTile(
                tensorA,
                tla::MakeCoord(nextBlockCoord.m() * L1_TILE_M, nextBlockCoord.k() * L1_TILE_K),
                tla::MakeShape(nextActualBlockShape.m(), nextActualBlockShape.k())
            );
            auto nextTensorBlockB = GetTile(
                tensorB,
                tla::MakeCoord(nextBlockCoord.k() * L1_TILE_K, nextBlockCoord.n() * L1_TILE_N),
                tla::MakeShape(nextActualBlockShape.k(), nextActualBlockShape.n())
            );

            // Compute block-scoped matrix multiply-add
            blockMmad(
                tensorBlockA, tensorBlockB, tensorBlockC, nextTensorBlockA, nextTensorBlockB, 
                actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock
            );
        }
    }

    // comm
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        shmemx_barrier_all_vec();
        if constexpr (!std::is_void_v<PaddingB>) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
            auto tensorWB = tla::MakeTensor(gmWB, params.layoutWB, Arch::PositionGM{});
            PaddingB paddingB(resource);
            paddingB(tensorWB, tensorB);
        }

        AscendC::GlobalTensor<ElementA> gmA;
        AscendC::GlobalTensor<ElementA> gmWA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        gmWA.SetGlobalBuffer((__gm__ ElementA *)params.ptrSymmetric);

        __gm__ int32_t * sigAddr = reinterpret_cast<__gm__ int32_t*>(params.ptrSignal);

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorSymmetric = tla::MakeTensor(gmWA, params.layoutWA, Arch::PositionGM{});
        
        BlockComm blockComm(resource);

        uint32_t rankId = shmem_my_pe();
        uint32_t rankSize = shmem_n_pes();

        uint32_t flagOffset = 1;
        for (uint32_t cocIdx = 1; cocIdx < rankSize; cocIdx = cocIdx * 2) {
            // 通信对象为最领近的、持有不同数据的rank
            uint32_t remoteId = ((rankId + cocIdx) % (cocIdx * 2)) + (rankId / (cocIdx * 2)) * (cocIdx * 2);
            uint32_t nextCocIdx = cocIdx * 2;
            uint32_t nextRemoteId = ((rankId + nextCocIdx) % (nextCocIdx * 2)) + (rankId / (nextCocIdx * 2)) * (nextCocIdx * 2);

            bool isFirstIter = (cocIdx == 1);
            // uint32_t symmOffset = (rankSize + rankId - cocIdx) % rankSize;
            uint32_t offset = ((rankId / cocIdx) % 2) == 0? 1: -1;
            uint32_t symmOffset = cocIdx * ((rankId / cocIdx) + offset);

            // rank本地地址
            auto tensorBlockA = GetTile(
                tensorA,
                tla::MakeCoord(static_cast<uint32_t>(0), static_cast<uint32_t>(0)),
                tla::MakeShape(params.problemShape.m(), params.problemShape.k())
            );

            // sm地址
            auto tensorBlockSymmetricForFirst = GetTile(
                tensorSymmetric,
                tla::MakeCoord(rankId * params.problemShape.m(), static_cast<uint32_t>(0)),
                tla::MakeShape(params.problemShape.m() * cocIdx, params.problemShape.k())
            );

            auto tensorBlockSymmetric = GetTile(
                tensorSymmetric,
                tla::MakeCoord(symmOffset * params.problemShape.m(), static_cast<uint32_t>(0)),
                tla::MakeShape(params.problemShape.m() * cocIdx, params.problemShape.k())
            );

            if (cocIdx != 1) { //等待远程rank将数据准备就绪,即上一轮完成
                shmem_signal_wait_until(sigAddr + ((flagOffset - 1) * 32), SHMEM_CMP_EQ, 1);
            }
            
            blockComm(tensorBlockSymmetric, tensorBlockSymmetricForFirst, tensorBlockA, sigAddr, rankId, rankSize, remoteId, isFirstIter);

            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            if ((AscendC::GetBlockIdx() == 0) && (cocIdx * 2 < rankSize)) { //通知下一轮的远程rank，数据准备就绪
                shmemx_signal_op(sigAddr + (flagOffset * 32), 1, SHMEM_SIGNAL_SET, nextRemoteId);
            }
            flagOffset += 1;
        }
        // 矩阵A数据准备就绪，开启mmad
        Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishAllGather);
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_ALLGATHER = 0;
    static constexpr Arch::FlagID RV_FLAG_AIV_FINISH_ALLGATHER = 1;
    Arch::CrossCoreFlagWithReverse<> flagAivFinishAllGather{FLAG_AIV_FINISH_ALLGATHER, RV_FLAG_AIV_FINISH_ALLGATHER};

    Arch::Resource<ArchTag> resource;
};

} // namespace Distributed::Kernel

#endif  // CATLASS_DISTRIBUTE_KERNEL_DIST_MATMUL_HPP