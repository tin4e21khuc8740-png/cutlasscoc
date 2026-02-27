#ifndef CATLASS_DISTRIBUTE_KERNEL_MATMUL_ALLREDUCE_HPP
#define CATLASS_DISTRIBUTE_KERNEL_MATMUL_ALLREDUCE_HPP

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
    class PaddingA,
    class PaddingB,
    class BlockComm_
>
class MatmulAllReduceTla {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    // using L1TileShape = typename BlockMmad::L1TileShape;

    using ElementA = typename BlockMmad::ElementA;      
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;
  
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutWA = typename BlockMmad::LayoutA;
    using LayoutWB = typename BlockMmad::LayoutB;

    template<class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template<>
    struct LayoutHelper<void> {
        using type = void;
    };
    using LayoutA = std::conditional_t<std::is_void_v<PaddingA>, LayoutWA, typename LayoutHelper<PaddingA>::type>; 
    using LayoutB = std::conditional_t<std::is_void_v<PaddingB>, LayoutWB, typename LayoutHelper<PaddingB>::type>;

    using BlockComm = BlockComm_;

    using BlockScheduler = BlockScheduler_;

    struct Params {
        // Data members
        GemmCoord problemShape;
        GemmCoord L1Shape;
        MatrixCoord commBlockShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWA;
        LayoutWA layoutWA;
        GM_ADDR ptrWB;
        LayoutWB layoutWB;
        GM_ADDR ptrSymmetric;     
        uint32_t rankSize;
        GM_ADDR ptrSignal; 

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_,
            GemmCoord const &L1Shape_, 
            MatrixCoord const &commBlockShape_,
            GM_ADDR ptrA_, LayoutA layoutA_, 
            GM_ADDR ptrB_, LayoutB layoutB_, 
            GM_ADDR ptrC_, LayoutC layoutC_, 
            GM_ADDR ptrWA_, LayoutWA layoutWA_,
            GM_ADDR ptrWB_, LayoutWB layoutWB_,
            GM_ADDR ptrSymmetric_, 
            uint32_t rankSize_, GM_ADDR ptrSignal_)
            : problemShape(problemShape_), L1Shape(L1Shape_), commBlockShape(commBlockShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_), ptrWA(ptrWA_), layoutWA(layoutWA_), ptrWB(ptrWB_), layoutWB(layoutWB_), ptrSymmetric(ptrSymmetric_), rankSize(rankSize_),
              ptrSignal(ptrSignal_){}
    };

    struct Arguments {
        GemmCoord problemShape;
        GemmCoord L1Shape;
        MatrixCoord commBlockShape;
        uint8_t *ptrA; LayoutA layoutA;
        uint8_t *ptrB; LayoutB layoutB;
        uint8_t *ptrC; LayoutC layoutC;
        uint8_t *ptrWA; LayoutWA layoutWA;
        uint8_t *ptrWB; LayoutWB layoutWB;
        uint8_t *ptrSymmetric; 
        uint32_t rankSize;
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
            args.L1Shape,
            args.commBlockShape,
            args.ptrA, args.layoutA,
            args.ptrB, args.layoutB,
            args.ptrC, args.layoutC,
            args.ptrWA, args.layoutWA,
            args.ptrWB, args.layoutWB,
            args.ptrSymmetric, 
            args.rankSize,
            args.ptrSignal
        };
        return params;
    }

    CATLASS_DEVICE
    MatmulAllReduceTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    // matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        // Catlass::Arch::CrossCoreWaitFlag(flagstart);
        // AscendC::PipeBarrier<PIPE_ALL>();
        // Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE2>();
        uint32_t rankSize = shmem_n_pes();
        int64_t rankId = shmem_my_pe();
        uint32_t aicId = AscendC::GetBlockIdx();
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();

        if (!std::is_void_v<PaddingA> || !std::is_void_v<PaddingB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        GemmCoord globalProblemShape{params.problemShape.m(), params.problemShape.n(), params.problemShape.k()};

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(params.L1Shape, resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrSymmetric);
        // gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        auto tensorA = tla::MakeTensor(gmA, params.layoutWA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutWB, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

        // 按commBlockShape对结果进行分块
        uint32_t mCocLoop = (params.problemShape.m() + params.commBlockShape.row() - 1) / params.commBlockShape.row();
        uint32_t nCocLoop = (params.problemShape.n() + params.commBlockShape.column() - 1) / params.commBlockShape.column();
        uint32_t CocLoop = mCocLoop * nCocLoop;
        for (uint32_t cocIdx = 0; cocIdx < CocLoop; cocIdx++) {  // 一次循环处理一个commBlockShape大小
    
            int32_t mcocIdx = cocIdx / nCocLoop;
            int32_t ncocIdx = cocIdx % nCocLoop;

            GemmCoord actualCocShape{params.commBlockShape.row(), params.commBlockShape.column(), params.problemShape.k()};
            if ((mcocIdx == mCocLoop - 1) && (params.problemShape.m() % params.commBlockShape.row() != 0)) {
                actualCocShape.m() = params.problemShape.m() % params.commBlockShape.row();
            }

            if ((ncocIdx == nCocLoop - 1) && (params.problemShape.n() % params.commBlockShape.column() != 0)) {
                actualCocShape.n() = params.problemShape.n() % params.commBlockShape.column();
            }

            BlockScheduler matmulBlockScheduler(actualCocShape, MakeCoord(params.L1Shape.m(), params.L1Shape.n()));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
            uint32_t begin_id = (cocIdx * coreLoops) % AscendC::GetBlockNum();

            Catlass::Arch::CrossCoreWaitFlag(flagstart);
            AscendC::PipeBarrier<PIPE_ALL>();
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE2>();

            for (uint32_t loopIdx = 0; loopIdx < coreLoops; loopIdx++) {
                if ((loopIdx + begin_id) % AscendC::GetBlockNum() != AscendC::GetBlockIdx()) {
                    continue;
                }
                
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                auto tensorBlockA = GetTile(
                    tensorA,
                    tla::MakeCoord(mcocIdx * params.commBlockShape.row() + blockCoord.m() * params.L1Shape.m(), blockCoord.k() * params.L1Shape.k()),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k())
                );
                
                auto tensorBlockB = GetTile(
                    tensorB,
                    tla::MakeCoord(blockCoord.k() * params.L1Shape.k(), ncocIdx * params.commBlockShape.column() + blockCoord.n() * params.L1Shape.n()),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n())
                );
                
                auto tensorBlockC = GetTile(
                    tensorC,
                    tla::MakeCoord(mcocIdx * params.commBlockShape.row() + blockCoord.m() * params.L1Shape.m(), ncocIdx * params.commBlockShape.column() + blockCoord.n() * params.L1Shape.n()),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n())
                );
                
                bool isFirstBlock = (loopIdx == (AscendC::GetBlockIdx() + AscendC::GetBlockNum() - begin_id) % AscendC::GetBlockNum());
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
                    tla::MakeCoord(mcocIdx * params.commBlockShape.row() + nextBlockCoord.m() * params.L1Shape.m(), nextBlockCoord.k() * params.L1Shape.k()),
                    tla::MakeShape(nextActualBlockShape.m(), nextActualBlockShape.k())
                );
                auto nextTensorBlockB = GetTile(
                    tensorB,
                    tla::MakeCoord(nextBlockCoord.k() * params.L1Shape.k(), ncocIdx * params.commBlockShape.column() + nextBlockCoord.n() * params.L1Shape.n()),
                    tla::MakeShape(nextActualBlockShape.k(), nextActualBlockShape.n())
                );
                
                blockMmad(
                    tensorBlockA, tensorBlockB, tensorBlockC, nextTensorBlockA, nextTensorBlockB, 
                    actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock
                );
                
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_FIX>();

            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAivFinishAllReduce);
        }     
        Catlass::Arch::CrossCoreWaitFlag(flagstart);
        AscendC::PipeBarrier<PIPE_ALL>();
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE2>();
    }

    // comm
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        shmemx_barrier_all_vec();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagstart);
        if constexpr (!std::is_void_v<PaddingA>) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
            auto tensorWA = tla::MakeTensor(gmWA, params.layoutWA, Arch::PositionGM{});
            PaddingA paddingA(resource);
            paddingA(tensorWA, tensorA);
        }

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
        if constexpr (!std::is_void_v<PaddingA> || !std::is_void_v<PaddingB>) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            AscendC::PipeBarrier<PIPE_ALL>();
            
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }

        AscendC::GlobalTensor<ElementC> gmC;
        AscendC::GlobalTensor<ElementC> gmWC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        gmWC.SetGlobalBuffer((__gm__ ElementC *)params.ptrSymmetric);

        __gm__ int32_t * sigAddr = reinterpret_cast<__gm__ int32_t*>(params.ptrSignal);

        uint32_t rankSize = shmem_n_pes();
        int64_t rankId = shmem_my_pe();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();

        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
        auto tensorSymmtric = tla::MakeTensor(gmWC, params.layoutC, Arch::PositionGM{});
        
        BlockComm blockComm(resource);
        
        // 按commBlockShape对结果进行分块
        uint32_t mCocLoop = (params.problemShape.m() + params.commBlockShape.row() - 1) / params.commBlockShape.row();
        uint32_t nCocLoop = (params.problemShape.n() + params.commBlockShape.column() - 1) / params.commBlockShape.column();
        uint32_t CocLoop = mCocLoop * nCocLoop;

        for (uint32_t cocIdx = 0; cocIdx < CocLoop; cocIdx++) {  // 一次循环处理一个commBlockShape大小
            int mcocIdx = cocIdx / nCocLoop;
            int ncocIdx = cocIdx % nCocLoop;

            MatrixCoord actualCommBlockShape{params.commBlockShape.row(), params.commBlockShape.column()};
            if ((mcocIdx == mCocLoop - 1) && (params.problemShape.m() % params.commBlockShape.row() != 0)) {
                actualCommBlockShape.row() = params.problemShape.m() % params.commBlockShape.row();
            }

            if ((ncocIdx == nCocLoop - 1) && (params.problemShape.n() % params.commBlockShape.column() != 0)) {
                actualCommBlockShape.column() = params.problemShape.n() % params.commBlockShape.column();
            }

            for(int i = 1; i < rankSize;i*=2){
                int targetid = rankId / (i * 2);
                targetid = targetid * (i * 2) + (rankId + i) % (i * 2);
                // int targetid = rankId + i;
                shmemx_signal_op(sigAddr + ((aivId * rankSize) + targetid) * 512, 0, SHMEM_SIGNAL_SET, rankId);
                shmemx_signal_op(sigAddr + ((aivNum * rankSize) + (aivId * rankSize) + targetid) * 512, 0, SHMEM_SIGNAL_SET, rankId);
                shmemx_signal_op(sigAddr + ((2 * aivNum * rankSize) + targetid) * 512, 0, SHMEM_SIGNAL_SET, rankId);
                // shmemx_signal_op(sigAddr + ((aivNum * rankSize) + (aicId * AscendC::GetSubBlockNum()) * rankSize + rankId) * 8, 0, SHMEM_SIGNAL_SET, i);
                // shmemx_signal_op(sigAddr + ((aivNum * rankSize) + (aicId * AscendC::GetSubBlockNum() + 1) * rankSize + rankId) * 8, 0, SHMEM_SIGNAL_SET, i);
            }
            // for(int i = 0; i < rankSize;i++){
            //     // int targetid = rankId / (i * 2);
            //     // targetid = targetid * (i * 2) + (rankId + i) % (i * 2);
            //     int targetid = rankId + i;
            //     shmemx_signal_op(sigAddr + ((aivId * rankSize) + targetid) * 512, 0, SHMEM_SIGNAL_SET, rankId);
            //     shmemx_signal_op(sigAddr + ((aivNum * rankSize) + (aivId * rankSize) + targetid) * 512, 0, SHMEM_SIGNAL_SET, rankId);
            //     shmemx_signal_op(sigAddr + ((2 * aivNum * rankSize) + targetid) * 512, 0, SHMEM_SIGNAL_SET, rankId);
            //     // shmemx_signal_op(sigAddr + ((aivNum * rankSize) + (aicId * AscendC::GetSubBlockNum()) * rankSize + rankId) * 8, 0, SHMEM_SIGNAL_SET, i);
            //     // shmemx_signal_op(sigAddr + ((aivNum * rankSize) + (aicId * AscendC::GetSubBlockNum() + 1) * rankSize + rankId) * 8, 0, SHMEM_SIGNAL_SET, i);
            // }
            // AscendC::PipeBarrier<PIPE_ALL>();

            Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(flagAivFinishAllReduce);
            // AscendC::PipeBarrier<PIPE_ALL>();

            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            AscendC::PipeBarrier<PIPE_ALL>();

            if(aivId < rankSize){
                // for(int i = 1; i < rankSize;i*=2){
                //     // shmemx_signal_op(sigAddr + (aivNum * rankSize + aivId * rankSize + rankId) * 8, 1, SHMEM_SIGNAL_SET, i);
                //     int targetid = rankId / (i * 2);
                //     targetid = targetid * (i * 2) + (rankId + i) % (i * 2);
                //     if(aivId == targetid){
                //         shmemx_signal_op(sigAddr + ((2 * aivNum * rankSize) + rankId) * 512, 1, SHMEM_SIGNAL_SET, targetid);
                //     }
                // }

                for(int i = 0; i < rankSize; i++){
                    // shmemx_signal_op(sigAddr + (aivNum * rankSize + aivId * rankSize + rankId) * 8, 1, SHMEM_SIGNAL_SET, i);
                    if(aivId == i){
                        shmemx_signal_op(sigAddr + ((2 * aivNum * rankSize) + rankId) * 512, 1, SHMEM_SIGNAL_SET, i);
                    }
                }
            }
            //  for(int i = 0; i < rankSize; i++){
            //     // shmemx_signal_op(sigAddr + (aivNum * rankSize + aivId * rankSize + rankId) * 8, 1, SHMEM_SIGNAL_SET, i);
            //         shmemx_signal_op(sigAddr + ((2 * aivNum * rankSize) + rankId) * 512, 1, SHMEM_SIGNAL_SET, i);
            // }

            // shmemx_barrier_all_vec();

            // rank本地地址
            auto tensorBlockC = GetTile(
                tensorC,
                tla::MakeCoord(mcocIdx * params.commBlockShape.row(), ncocIdx * params.commBlockShape.column()),
                tla::MakeShape(actualCommBlockShape.row(), actualCommBlockShape.column())
            );

            // sm地址
            auto tensorBlockSymmtric = GetTile(
                tensorSymmtric,
                tla::MakeCoord(mcocIdx * params.commBlockShape.row(), ncocIdx * params.commBlockShape.column()),
                tla::MakeShape(actualCommBlockShape.row(), actualCommBlockShape.column())
            );

            // shmemx_barrier_all_vec();
            if(actualCommBlockShape.row() <= (aivNum / 2)){
                int sp = aivNum / actualCommBlockShape.row();
                blockComm(tensorBlockC, tensorBlockSymmtric, sigAddr, sp);
            } else{
                blockComm(tensorBlockC, tensorBlockSymmtric, sigAddr);
            }

            // shmemx_barrier_all_vec();
            AscendC::PipeBarrier<PIPE_ALL>();
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagstart);
        }
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_ALLREDUCE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIV_FINISH_ALLREDUCE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAivFinishAllReduce{FLAG_AIV_FINISH_ALLREDUCE, RV_FLAG_AIV_FINISH_ALLREDUCE};

    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 2;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};

    static constexpr Arch::FlagID FLAG_START_STORE = 3;
    Arch::CrossCoreFlag flagstart{FLAG_START_STORE};

    Arch::Resource<ArchTag> resource;
};

} // namespace Distributed::Kernel

#endif  // CATLASS_DISTRIBUTE_KERNEL_DIST_MATMUL_HPP