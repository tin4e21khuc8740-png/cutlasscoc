#ifndef CATLASS_DISTRIBUTE_BLOCK_BLOCK_ALLREDUCE_HPP
#define CATLASS_DISTRIBUTE_BLOCK_BLOCK_ALLREDUCE_HPP

#include "catlass/catlass.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/distributed/dispatch_policy.hpp"
// #include "catlass/distributed/tile/copy_local_gm_to_shmem.hpp"
#include "catlass/distributed/tile/copy_shmem_to_local_gm.hpp"
#include "catlass/distributed/tile/copy_local_ub_to_shmem.hpp"
#include "catlass/distributed/tile/copy_shmem_to_local_ub.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "shmem_api.h"

#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_rma.h"
#include "host/shmem_host_team.h"

namespace Catlass::Distributed::Block {

template <
    class TensorIn_,
    class TensorOut_
>
struct BlockComm <
    CommAtlasA2AllReduce, 
    TensorIn_,
    TensorOut_
> {
public:
    using ArchTag = CommAtlasA2AllReduce::ArchTag;

    using TensorIn = TensorIn_;
    using Element = typename TensorIn::Element;
    using LayoutIn = typename TensorIn::Layout;
    using TensorOut = TensorOut_;
    using LayoutOut = typename TensorOut::Layout;

    // using LayoutWorkspace = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
    // using TensorWorkspace = tla::Tensor<AscendC::LocalTensor<Element>, LayoutWorkspace, AscendC::TPosition::VECCALC>;

    // using CopyLocalGmToShmem = Catlass::Distributed::Tile::CopyLocalGmToShmem<ArchTag, TensorOut, TensorIn, TensorWorkspace>;
    // CopyLocalGmToShmem copyLmemToShmem;

    // using CopyShmemToLocalGm = Catlass::Distributed::Tile::CopyShmemToLocalGm<ArchTag, TensorOut, TensorIn, TensorWorkspace>;
    // CopyShmemToLocalGm copyShmemToLmem;

    using LayoutInner = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
    using TensorInnerUb = tla::Tensor<AscendC::LocalTensor<Element>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;
    using TensorInnerSrcGm = tla::Tensor<AscendC::GlobalTensor<Element>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorInnerDstGm = tla::Tensor<AscendC::GlobalTensor<Element>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    using CopyShmemToLocalGm = Catlass::Distributed::Tile::CopyShmemToLocalGm<ArchTag, TensorOut, TensorIn, TensorInnerUb>;
    CopyShmemToLocalGm copyShmemToLmem;
    
    using CopyUb2GM = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorInnerUb, TensorInnerDstGm>;
    using CopyShmemToLocalUb = Catlass::Distributed::Tile::CopyShmemToLocalUb<ArchTag, TensorInnerUb, TensorInnerSrcGm>;
    using CopyLocalUbToShmem = Catlass::Distributed::Tile::CopyLocalUbToShmem<ArchTag, TensorInnerDstGm, TensorInnerUb>;

    CopyUb2GM copyUb2Gm;
    CopyShmemToLocalUb copyShmemToLocalUb;
    CopyLocalUbToShmem copyLocalUbToShmem;

    CATLASS_DEVICE
    BlockComm(Arch::Resource<ArchTag> &resource)
    {
        uint64_t bufferOffset = 0;
        // BUFFER_NUM
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            tmpBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset);
            bufferOffset += tmpBufferSize;
        }
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            addBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset);
            bufferOffset += tmpBufferSize;
        }
    }

    CATLASS_DEVICE
    ~BlockComm() {}

    template <class TensorOut, class TensorIn>
    CATLASS_DEVICE
    void operator()(TensorOut &tensorDst, TensorIn &tensorSrc, __gm__ int32_t *signal)
    {
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint32_t aicoreId = AscendC::GetBlockIdx() / 2;
        uint32_t aivSubId = AscendC::GetSubBlockIdx();

        int64_t rankId = shmem_my_pe();
        int64_t rankSize = shmem_n_pes();

        uint32_t tilesNum = tla::get<0>(tensorSrc.shape());
        uint32_t tileLen = tla::get<1>(tensorSrc.shape());
        uint32_t roundTileLen = RoundUp<BYTE_PER_BLK / sizeof(Element)>(tileLen);

        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        uint32_t mIdx = aivId * tilesPerAiv;
        if(aivId >= tileRemain) {
            mIdx += tileRemain;
        }

        uint32_t coreLoops{ 0 };
        uint32_t tilesPerLoop = COMPUTE_LENGTH / roundTileLen;
        coreLoops = (tilesPerAiv + tilesPerLoop - 1) / tilesPerLoop;
        
        // workspaceTensor中的layout和shape信息无效
        auto layoutWorkspace = MakeLayout(
            tla::MakeShape(static_cast<uint32_t>(1), static_cast<uint32_t>(1)),
            tla::MakeStride(static_cast<int64_t>(1), tla::Int<1>{})
        );
        auto tensorWorkspace = tla::MakeTensor(
            tmpBuffer[0],
            layoutWorkspace,
            Arch::PositionUB{}
        );

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {

            uint32_t tileIdx = loopIdx * tilesPerLoop;
            uint32_t actualTilesNum = tilesPerLoop;
           
            if (tilesPerAiv - tileIdx < tilesPerLoop) {
                actualTilesNum = tilesPerAiv - tileIdx;
            }
        
            auto offset = tla::MakeCoord(mIdx + tileIdx, static_cast<uint32_t>(0));

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
            auto tensorTileSymmSrc = GetTile(
                tensorSrc,
                offset,
                tla::MakeShape(actualTilesNum, tileLen)
            );

            auto layoutUb = MakeLayout(
                tla::MakeShape(actualTilesNum, tileLen),
                tla::MakeStride(static_cast<int64_t>(roundTileLen), tla::Int<1>{})
            );
            auto tensoraddUb = tla::MakeTensor(addBuffer[0], layoutUb, Arch::PositionUB{});
            // AscendC::Duplicate<half>(addBuffer[bufferIndex], (half)0.0, (actualTilesNum * roundTileLen));
            // shmem_signal_wait_until(signal + ((aivNum * rankSize) + (aivId * rankSize) + rankId) * 8, SHMEM_CMP_EQ, 1);
            shmem_signal_wait_until(signal + ((2 * aivNum * rankSize) + rankId) * 512, SHMEM_CMP_EQ, 1);
            copyShmemToLocalUb(tensoraddUb, tensorTileSymmSrc, rankId, eventIds[0]);
            
            for(int i = 1; i < rankSize;i*=2){
                auto tensorUb = tla::MakeTensor(tmpBuffer[0], layoutUb, Arch::PositionUB{});
                int targetid = rankId / (i * 2);
                targetid = targetid * (i * 2) + (rankId + i) % (i * 2);
                int nextid = -1;
                if((i * 2) < rankSize){
                    nextid = rankId / (i * 4);
                    nextid = nextid * (i * 4) + (rankId + (i * 2)) % (i * 4);
                }
                
                shmem_signal_wait_until(signal + ((2 * aivNum * rankSize) + targetid) * 512, SHMEM_CMP_EQ, 1);
                if(i > 1){
                    shmem_signal_wait_until(signal + ((aivId * rankSize) + targetid) * 512, SHMEM_CMP_EQ, 1);
                }
                copyShmemToLocalUb(tensorUb, tensorTileSymmSrc, targetid, eventIds[0]);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIds[0]);
                shmemx_signal_op(signal + ((aivNum * rankSize) + (aivId * rankSize) + rankId) * 512, 1, SHMEM_SIGNAL_SET, targetid);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIds[0]);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);
                AscendC::Add(addBuffer[0], tmpBuffer[0], addBuffer[0], (actualTilesNum * roundTileLen));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                if(nextid != -1){
                    shmem_signal_wait_until(signal + ((aivNum * rankSize) + (aivId * rankSize) + targetid) * 512, SHMEM_CMP_EQ, 1);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);
                    // copyUb2Gm(tensorTileSymmSrc, tensoraddUb);
                    copyLocalUbToShmem(tensorTileSymmSrc, tensoraddUb, rankId, eventIds[0]);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventIds[0]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventIds[0]);
                    shmemx_signal_op(signal + (aivId * rankSize + rankId) * 512, 1, SHMEM_SIGNAL_SET, nextid);
                }
            }
            // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
            // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[1]);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);

            auto tensorTileA = GetTile(
                tensorDst,
                offset,
                tla::MakeShape(actualTilesNum, tileLen)
            );
            copyUb2Gm(tensorTileA, tensoraddUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
    }

    template <class TensorOut, class TensorIn>
    CATLASS_DEVICE
    void operator()(TensorOut &tensorDst, TensorIn &tensorSrc, __gm__ int32_t *signal, int spl)
    {
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint32_t aicoreId = AscendC::GetBlockIdx() / 2;
        uint32_t aivSubId = AscendC::GetSubBlockIdx();

        int64_t rankId = shmem_my_pe();
        int64_t rankSize = shmem_n_pes();

        int split = spl;
        uint32_t tilesNum = tla::get<0>(tensorSrc.shape()) * split;
        uint32_t tileLen = (tla::get<1>(tensorSrc.shape()) + split - 1) / split;
        uint32_t roundTileLen = RoundUp<BYTE_PER_BLK / sizeof(Element)>(tileLen);
        split = (tla::get<1>(tensorSrc.shape()) + roundTileLen - 1) / roundTileLen;
        tilesNum = tla::get<0>(tensorSrc.shape()) * split;
        tileLen = (tla::get<1>(tensorSrc.shape()) + split - 1) / split;
        roundTileLen = RoundUp<BYTE_PER_BLK / sizeof(Element)>(tileLen);

        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        
        // workspaceTensor中的layout和shape信息无效
        auto layoutWorkspace = MakeLayout(
            tla::MakeShape(static_cast<uint32_t>(1), static_cast<uint32_t>(1)),
            tla::MakeStride(static_cast<int64_t>(1), tla::Int<1>{})
        );
        auto tensorWorkspace = tla::MakeTensor(
            tmpBuffer[0],
            layoutWorkspace,
            Arch::PositionUB{}
        );
        
        int waitflag = 0;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        for(int splitloopid = 0; splitloopid < tilesNum; splitloopid++){
            if((splitloopid % aivNum) != aivId){
                continue;
            }

            uint32_t midx = splitloopid / split;
            uint32_t sidx = splitloopid % split;

            uint32_t actualTilesLen = roundTileLen;
            if(sidx == (split - 1)){
                actualTilesLen = tla::get<1>(tensorSrc.shape()) - (sidx * roundTileLen);
            }
            
            auto offset = tla::MakeCoord(midx, sidx * roundTileLen);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
            auto tensorTileSymmSrc = GetTile(
                tensorSrc,
                offset,
                tla::MakeShape(1, actualTilesLen)
            );

            auto layoutUb = MakeLayout(
                tla::MakeShape(1, actualTilesLen),
                tla::MakeStride(static_cast<int64_t>(RoundUp<BYTE_PER_BLK / sizeof(Element)>(tla::get<1>(tensorSrc.shape()))), tla::Int<1>{})
            );
            auto tensoraddUb = tla::MakeTensor(addBuffer[0], layoutUb, Arch::PositionUB{});
            // AscendC::Duplicate<half>(addBuffer[bufferIndex], (half)0.0, (actualTilesNum * roundTileLen));
            // shmem_signal_wait_until(signal + ((aivNum * rankSize) + (aivId * rankSize) + rankId) * 8, SHMEM_CMP_EQ, 1);
            shmem_signal_wait_until(signal + ((2 * aivNum * rankSize) + rankId) * 512, SHMEM_CMP_EQ, 1);
            copyShmemToLocalUb(tensoraddUb, tensorTileSymmSrc, rankId, eventIds[0]);
        
            waitflag++;
            for(int i = 1; i < rankSize;i*=2){
                auto tensorUb = tla::MakeTensor(tmpBuffer[0], layoutUb, Arch::PositionUB{});
                int targetid = rankId / (i * 2);
                targetid = targetid * (i * 2) + (rankId + i) % (i * 2);
                int nextid = -1;
                if((i * 2) < rankSize){
                    nextid = rankId / (i * 4);
                    nextid = nextid * (i * 4) + (rankId + (i * 2)) % (i * 4);
                }
                
                shmem_signal_wait_until(signal + ((2 * aivNum * rankSize) + targetid) * 512, SHMEM_CMP_EQ, 1);
                if(i > 1){
                    shmem_signal_wait_until(signal + ((aivId * rankSize) + targetid) * 512, SHMEM_CMP_EQ, waitflag);
                }
                copyShmemToLocalUb(tensorUb, tensorTileSymmSrc, targetid, eventIds[0]);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIds[0]);
                shmemx_signal_op(signal + ((aivNum * rankSize) + (aivId * rankSize) + rankId) * 512, 1, SHMEM_SIGNAL_ADD, targetid);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIds[0]);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);
                AscendC::Add(addBuffer[0], tmpBuffer[0], addBuffer[0], (actualTilesLen));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                if(nextid != -1){
                    shmem_signal_wait_until(signal + ((aivNum * rankSize) + (aivId * rankSize) + targetid) * 512, SHMEM_CMP_EQ, waitflag);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);
                    // copyUb2Gm(tensorTileSymmSrc, tensoraddUb);
                    copyLocalUbToShmem(tensorTileSymmSrc, tensoraddUb, rankId, eventIds[0]);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventIds[0]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventIds[0]);
                    shmemx_signal_op(signal + (aivId * rankSize + rankId) * 512, 1, SHMEM_SIGNAL_ADD, nextid);
                }
            }
            // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
            // AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[1]);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);

            auto tensorTileA = GetTile(
                tensorDst,
                offset,
                tla::MakeShape(1, actualTilesLen)
            );
            // AscendC::PipeBarrier<PIPE_ALL>();
            copyUb2Gm(tensorTileA, tensoraddUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        }
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
    } 

private:
    // Params params;
    static const uint32_t BUFFER_NUM = 1;
    uint32_t tmpBufferSize = 96 * 1024; // 96KB
    uint32_t COMPUTE_LENGTH = 96 * 1024 / sizeof(Element); // 48K

    AscendC::GlobalTensor<Element> gmA;
    AscendC::GlobalTensor<Element> gmWA;
    AscendC::LocalTensor<Element> tmpBuffer[BUFFER_NUM];
    AscendC::LocalTensor<Element> addBuffer[BUFFER_NUM];
    // AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0}; //int32_t eventIDSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
    uint32_t bufferIndex{ 0 };
    uint32_t addbufferIndex{ 0 };
};

} // namespace Catlass::Distributed::Block

#endif  // CATLASS_DISTRIBUTE_BLOCK_BLOCK_ALLREDUCE_HPP