#ifndef CATLASS_DISTRIBUTE_KERNEL_MATMUL_REDUCESCATTER_HPP
#define CATLASS_DISTRIBUTE_KERNEL_MATMUL_REDUCESCATTER_HPP

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
class MatmulReduceScatterTla {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;

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
        GM_ADDR ptrSyncFlags;
        uint32_t rankSize;

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
            GM_ADDR ptrSyncFlags_, 
            uint32_t rankSize_)
            : problemShape(problemShape_), L1Shape(L1Shape_), commBlockShape(commBlockShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_), ptrWA(ptrWA_), layoutWA(layoutWA_), ptrWB(ptrWB_), layoutWB(layoutWB_), ptrSymmetric(ptrSymmetric_), ptrSyncFlags(ptrSyncFlags_), rankSize(rankSize_) {}
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
        uint8_t *ptrSyncFlags;
        uint32_t rankSize;
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
            args.ptrSyncFlags,
            args.rankSize
        };
        return params;
    }

    CATLASS_DEVICE
    MatmulReduceScatterTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    // matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        uint32_t rankSize = shmem_n_pes();
        int64_t rankId = shmem_my_pe();
        if (!std::is_void_v<PaddingA> || !std::is_void_v<PaddingB>) {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishPadding);
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
        auto tensorA = tla::MakeTensor(gmA, params.layoutWA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutWB, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});

        GemmCoord actualCocShape{params.problemShape.m(), params.problemShape.n(), params.problemShape.k()};
        BlockScheduler matmulBlockScheduler(actualCocShape, MakeCoord(params.L1Shape.m(), params.L1Shape.n()));

        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        for (uint32_t loopIdx = 0; loopIdx < coreLoops; loopIdx++) {        
            if (loopIdx % AscendC::GetBlockNum() != AscendC::GetBlockIdx()) {
                continue;
            }
    
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            auto tensorBlockA = GetTile(
                tensorA,
                tla::MakeCoord(blockCoord.m() * params.L1Shape.m(), blockCoord.k() * params.L1Shape.k()),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k())
            );
    
            auto tensorBlockB = GetTile(
                tensorB,
                tla::MakeCoord(blockCoord.k() * params.L1Shape.k(), blockCoord.n() * params.L1Shape.n()),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n())
            );
    
            auto tensorBlockC = GetTile(
                tensorC,
                tla::MakeCoord(blockCoord.m() * params.L1Shape.m(), blockCoord.n() * params.L1Shape.n()),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n())
            );

            bool isFirstBlock = (loopIdx == (AscendC::GetBlockIdx() + AscendC::GetBlockNum()) % AscendC::GetBlockNum());
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
                tla::MakeCoord(nextBlockCoord.m() * params.L1Shape.m(), nextBlockCoord.k() * params.L1Shape.k()),
                tla::MakeShape(nextActualBlockShape.m(), nextActualBlockShape.k())
            );
            auto nextTensorBlockB = GetTile(
                tensorB,
                tla::MakeCoord(nextBlockCoord.k() * params.L1Shape.k(), nextBlockCoord.n() * params.L1Shape.n()),
                tla::MakeShape(nextActualBlockShape.k(), nextActualBlockShape.n())
            );
    
            blockMmad(
                tensorBlockA, tensorBlockB, tensorBlockC, nextTensorBlockA, nextTensorBlockB, 
                actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock
            );
        }
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_FIX>();
        Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishMatmul);
    }

    // comm
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        shmemx_barrier_all_vec();
        int64_t rankId = shmem_my_pe();
        uint32_t rankSize = shmem_n_pes();
        
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
            // AscendC::PipeBarrier<PIPE_ALL>();
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }
        AscendC::GlobalTensor<ElementC> gmC;
        AscendC::GlobalTensor<ElementC> gmWC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        gmWC.SetGlobalBuffer((__gm__ ElementC *)params.ptrSymmetric);

        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
        auto tensorSymmtric = tla::MakeTensor(gmWC, params.layoutC, Arch::PositionGM{});
        
        BlockComm blockComm(resource);

        Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishMatmul);
        shmemx_barrier_all_vec();

        MatrixCoord actualCommBlockShape{params.commBlockShape.row(), params.commBlockShape.column()};

            auto tensorBlockC = GetTile(
                tensorSymmtric,
                tla::MakeCoord(rankId * params.commBlockShape.row(), 0),  
                tla::MakeShape(actualCommBlockShape.row(), actualCommBlockShape.column())
            );           
            auto tensorBlockGm = GetTile(
                tensorC,
                tla::MakeCoord(rankId * params.commBlockShape.row(), 0),                 
                tla::MakeShape(actualCommBlockShape.row(), actualCommBlockShape.column())
            );
        blockComm(tensorBlockGm, tensorBlockC);
        // shmemx_barrier_all_vec();
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_REDUCESCATTER = 0;
    static constexpr Arch::FlagID RV_FLAG_AIV_FINISH_REDUCESCATTER = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishMatmul{FLAG_AIV_FINISH_REDUCESCATTER, RV_FLAG_AIV_FINISH_REDUCESCATTER};
    // Arch::CrossCoreFlag flagAicFinishMatmul{FLAG_AIV_FINISH_REDUCESCATTER};
    //  Arch::CrossCoreFlag flagAivFinishAllReducepong{RV_FLAG_AIV_FINISH_REDUCESCATTER};
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 2;
    static constexpr Arch::FlagID RV_FLAG_AIV_FINISH_STORE = 3; 
    Arch::CrossCoreFlagWithReverse<> flagAivFinishPadding{FLAG_AIV_FINISH_STORE, RV_FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Distributed::Kernel

#endif  // CATLASS_DISTRIBUTE_KERNEL_DIST_MATMUL_HPP