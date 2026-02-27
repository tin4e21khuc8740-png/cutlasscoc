#ifndef CATLASS_DISTRIBUTE_BLOCK_BLOCK_REDUCESCATTER_HPP
#define CATLASS_DISTRIBUTE_BLOCK_BLOCK_REDUCESCATTER_HPP

#include "catlass/catlass.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/distributed/dispatch_policy.hpp"
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
    CommAtlasA2ReduceScatter, 
    TensorIn_,
    TensorOut_
> {
public:
    using ArchTag = CommAtlasA2ReduceScatter::ArchTag;
    using TensorIn = TensorIn_;
    using Element = typename TensorIn::Element;
    using TensorOut = TensorOut_;
    using LayoutIn = typename TensorIn::Layout;
    using LayoutOut = typename TensorOut::Layout;

    using LayoutInner = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
    using TensorInnerUb = tla::Tensor<AscendC::LocalTensor<Element>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;
    using TensorInnerSrcGm = tla::Tensor<AscendC::GlobalTensor<Element>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorInnerDstGm = tla::Tensor<AscendC::GlobalTensor<Element>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    using CopyUb2Gm = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorInnerUb, TensorInnerDstGm>;
    using CopyShmemToLocalUb = Catlass::Distributed::Tile::CopyShmemToLocalUb<ArchTag, TensorInnerUb, TensorInnerSrcGm>;
    using CopyLocalUbToShmem = Catlass::Distributed::Tile::CopyLocalUbToShmem<ArchTag, TensorInnerDstGm, TensorInnerUb>;

    CopyUb2Gm copyUb2Gm;
    CopyShmemToLocalUb copyShmemToLocalUb;
    CopyLocalUbToShmem copyLocalUbToShmem;

    CATLASS_DEVICE
    BlockComm(Arch::Resource<ArchTag> &resource)
    {
        uint64_t bufferOffset = 0;

        for (uint32_t i = 0; i < 2; i++) {
            tmpBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset);
            bufferOffset += tmpBufferSize;
        }

        addBuffer[0] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset);
        bufferOffset += tmpBufferSize;
    }

    CATLASS_DEVICE
    ~BlockComm() {}

    template <typename T_Dst, typename T_Src>
    CATLASS_DEVICE
    void operator()(T_Dst &tensorDst, T_Src &tensorSrc)
    {
        int64_t rankId = shmem_my_pe();
        uint32_t rankSize = shmem_n_pes();

        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint32_t tilesNum = tla::get<0>(tensorSrc.shape());
        uint32_t tileLen = tla::get<1>(tensorSrc.shape());
        // uint32_t roundTileLen = RoundUp<BYTE_PER_BLK / sizeof(Element)>(tileLen);
        uint32_t roundTileLen = RoundUp<512 / sizeof(Element)>(tileLen);
        
        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        uint32_t mIdx = aivId * tilesPerAiv + (aivId >= tileRemain ? tileRemain : 0);
        uint32_t tilesPerLoop = COMPUTE_LENGTH / roundTileLen;
        uint32_t coreloops = (tilesPerAiv + tilesPerLoop - 1) / tilesPerLoop;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);

        for (uint32_t loopIdx = 0; loopIdx < coreloops; ++loopIdx) {
            uint32_t tileIdx = loopIdx * tilesPerLoop;
            uint32_t actualTilesNum = (tilesPerAiv - tileIdx < tilesPerLoop) ? (tilesPerAiv - tileIdx) : tilesPerLoop;
            const uint32_t computeLength = actualTilesNum * roundTileLen;
            auto offset = tla::MakeCoord(mIdx + tileIdx, static_cast<uint32_t>(0));
            
            auto tensorTileSymmSrc = GetTile(tensorSrc, offset, tla::MakeShape(actualTilesNum, tileLen));
            auto layoutUb = MakeLayout(tla::MakeShape(actualTilesNum, tileLen), tla::MakeStride(static_cast<int64_t>(roundTileLen), tla::Int<1>{}));
            
            auto tensorAddUb = tla::MakeTensor(addBuffer[0], layoutUb, Arch::PositionUB{});
            auto tensorTmpUb0 = tla::MakeTensor(tmpBuffer[0], layoutUb, Arch::PositionUB{});
            auto tensorTmpUb1 = tla::MakeTensor(tmpBuffer[1], layoutUb, Arch::PositionUB{});

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]); 

            copyShmemToLocalUb(tensorAddUb, tensorTileSymmSrc, rankId, eventIds[0]);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]); 

            uint32_t remoteCount = rankSize - 1; 

            if (remoteCount > 0) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[1]);

                uint32_t midx_0 = (rankId + 1) % rankSize;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]); 
                copyShmemToLocalUb(tensorTmpUb0, tensorTileSymmSrc, midx_0, eventIds[0]);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]); 

                uint32_t k = 0;
                for (; k < remoteCount - 1; k += 2) {
                    uint32_t midx_next = (rankId + 1 + (k + 1)) % rankSize;
                    
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[1]);
                    copyShmemToLocalUb(tensorTmpUb1, tensorTileSymmSrc, midx_next, eventIds[1]);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[1]);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);
                    AscendC::Add(addBuffer[0], tmpBuffer[0], addBuffer[0], computeLength);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]); // Buffer 0 Free

                    if (k + 2 < remoteCount) {
                        uint32_t midx_next_next = (rankId + 1 + (k + 2)) % rankSize;
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                        copyShmemToLocalUb(tensorTmpUb0, tensorTileSymmSrc, midx_next_next, eventIds[0]);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);
                    }

                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[1]);
                    AscendC::Add(addBuffer[0], tmpBuffer[1], addBuffer[0], computeLength);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[1]); // Buffer 1 Free
                }
                if (k < remoteCount) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[0]);
                    AscendC::Add(addBuffer[0], tmpBuffer[0], addBuffer[0], computeLength);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                }
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[1]);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIds[0]);

            auto tensorTileGmDst = GetTile(tensorDst, offset, tla::MakeShape(actualTilesNum, tileLen));
            copyUb2Gm(tensorTileGmDst, tensorAddUb);
            
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        }
        
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
    }

private:
    uint32_t tmpBufferSize = 64 * 1024; 
    uint32_t COMPUTE_LENGTH = 64 * 1024 / sizeof(Element);
    
    AscendC::LocalTensor<Element> tmpBuffer[2];
    AscendC::LocalTensor<Element> addBuffer[1];
    
    AscendC::TEventID eventIds[2] = {EVENT_ID0, EVENT_ID1};
};

} // namespace Catlass::Distributed::Block

#endif  // CATLASS_DISTRIBUTE_BLOCK_BLOCK_REDUCESCATTER_HPP