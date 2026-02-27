#ifndef CATLASS_DISTRIBUTE_BLOCK_BLOCK_ALLGATHER_HPP
#define CATLASS_DISTRIBUTE_BLOCK_BLOCK_ALLGATHER_HPP

#include "catlass/catlass.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/distributed/dispatch_policy.hpp"
#include "catlass/distributed/tile/copy_local_ub_to_shmem.hpp"
#include "catlass/distributed/tile/copy_shmem_to_local_ub.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "shmem_api.h"

namespace Catlass::Distributed::Block {

template <
    class TensorIn_,
    class TensorOut_
>
struct BlockComm <
    CommAtlasA2AllGather, 
    TensorIn_,
    TensorOut_
> {
public:
    using ArchTag = CommAtlasA2AllGather::ArchTag;

    using TensorIn = TensorIn_;
    using TensorOut = TensorOut_;
    using Element = typename TensorIn::Element;
    using LayoutIn = typename TensorIn::Layout;
    using LayoutOut = typename TensorOut::Layout;

    using LayoutInner = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
    using TensorInnerUb = tla::Tensor<AscendC::LocalTensor<Element>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;
    using TensorInnerSrcGm = tla::Tensor<AscendC::GlobalTensor<Element>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorInnerDstGm = tla::Tensor<AscendC::GlobalTensor<Element>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    using CopyGm2Ub = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorInnerSrcGm, TensorInnerUb>;

    using CopyShmemToLocalUb = Catlass::Distributed::Tile::CopyShmemToLocalUb<ArchTag, TensorInnerUb, TensorInnerSrcGm>;
    using CopyLocalUbToShmem = Catlass::Distributed::Tile::CopyLocalUbToShmem<ArchTag, TensorInnerDstGm, TensorInnerUb>;

    CopyGm2Ub copyGm2Ub;
    CopyShmemToLocalUb copyShmemToLocalUb;
    CopyLocalUbToShmem copyLocalUbToShmem;

    CATLASS_DEVICE
    BlockComm(Arch::Resource<ArchTag> &resource)
    {
        uint64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            tmpBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset);
            bufferOffset += tmpBufferSize;
        }
    }

    CATLASS_DEVICE
    ~BlockComm() {}

    template<class Tensor>
    CATLASS_DEVICE
    auto GetPaddingTensor(Tensor const &tensor)
    {
        if constexpr (std::is_same_v<typename Tensor::Layout, LayoutInner>) {
            return tensor;
        } else {
            auto shape = tla::MakeShape(tla::get<1>(tensor.shape()), tla::get<0>(tensor.shape()));
            auto stride = tla::MakeStride(tla::get<1>(tensor.stride()), tla::get<0>(tensor.stride()));
            return tla::MakeTensor(tensor.data(), MakeLayout(shape, stride), Arch::PositionGM{});
        }
    }

    template <class TensorOut, class TensorIn>
    CATLASS_DEVICE
    void operator()(TensorOut &tensorSymm, TensorOut &tensorSymmForFirst,
                    TensorIn const& tensorA, __gm__ int32_t * sigAddr,
                    uint32_t rankId, uint32_t rankSize, 
                    uint32_t remoteId, bool isFirstIter)
    {
        // 行优先保持原本数据布局，列优先交换两维
        auto paddingTensorSrc = GetPaddingTensor(tensorA);
        auto paddingTensorDst = GetPaddingTensor(tensorSymm);

        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint32_t aivSubId = AscendC::GetSubBlockIdx();

        // uint32_t tilesNum = tla::get<0>(paddingTensorSrc.shape());
        // uint32_t tileLen = tla::get<1>(paddingTensorSrc.shape());
        uint32_t tilesNum = tla::get<0>(paddingTensorDst.shape());
        uint32_t tileLen = tla::get<1>(paddingTensorDst.shape());
        // if (aivId == 0 && isFirstIter) {
        //     AscendC::printf("!!!!!!!!!!!!!%d  %d\n", tilesNum, tileLen);
        // }
        uint32_t roundTileLen = RoundUp<BYTE_PER_BLK / sizeof(Element)>(tileLen); // 32B对齐

        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        uint32_t mIdx = aivId * tilesPerAiv;
        if(aivId >= tileRemain) {
            mIdx += tileRemain;
        }
        MatrixCoord blockOffset(mIdx, 0);

        uint32_t coreLoops{ 0 };
        uint32_t tilesPerLoop = COMPUTE_LENGTH / roundTileLen;
        coreLoops = (tilesPerAiv + tilesPerLoop - 1) / tilesPerLoop;

        if (isFirstIter) { //第一轮从本地的gmA读取并padding到本rank的sm上
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx * tilesPerLoop;
                uint32_t actualTilesNum = tilesPerLoop;
                if (tilesPerAiv - tileIdx < tilesPerLoop) {
                    actualTilesNum = tilesPerAiv - tileIdx;
                }
                auto offset = tla::MakeCoord(mIdx + tileIdx, static_cast<uint32_t>(0));

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                auto tensorTileA = GetTile(
                    paddingTensorSrc,
                    offset,
                    tla::MakeShape(actualTilesNum, tileLen)
                );

                auto layoutUb = MakeLayout(
                    tla::MakeShape(actualTilesNum, tileLen),
                    tla::MakeStride(static_cast<int64_t>(roundTileLen), tla::Int<1>{})
                );
                auto tensorUb = tla::MakeTensor(tmpBuffer[bufferIndex], layoutUb, Arch::PositionUB{});

                copyGm2Ub(tensorUb, tensorTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                auto tensorTileSymmDst = GetTile(
                    tensorSymmForFirst,
                    offset,
                    tla::MakeShape(actualTilesNum, tileLen)
                );

                copyLocalUbToShmem(tensorTileSymmDst,
                                   tensorUb,
                                   rankId,
                                   eventIds[bufferIndex]);
                
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);

            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            if (aivId == 0) { //标志远程rank，数据准备就绪
                shmemx_signal_op(sigAddr, 1, SHMEM_SIGNAL_SET, remoteId);
            }

            //第一轮等待远程rank将数据搬到远程sm
            shmem_signal_wait_until(sigAddr, SHMEM_CMP_EQ, 1);
            // shmemx_barrier_all_vec();
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
        for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
            uint32_t tileIdx = loopIdx * tilesPerLoop;
            uint32_t actualTilesNum = tilesPerLoop;
            if (tilesPerAiv - tileIdx < tilesPerLoop) {
                actualTilesNum = tilesPerAiv - tileIdx;
            }
            auto offset = tla::MakeCoord(mIdx + tileIdx, static_cast<uint32_t>(0));

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
            auto tensorTileSymmSrc = GetTile(
                paddingTensorDst,
                offset,
                tla::MakeShape(actualTilesNum, tileLen)
            );

            auto layoutUb = MakeLayout(
                tla::MakeShape(actualTilesNum, tileLen),
                tla::MakeStride(static_cast<int64_t>(roundTileLen), tla::Int<1>{})
            );
            auto tensorUb = tla::MakeTensor(tmpBuffer[bufferIndex], layoutUb, Arch::PositionUB{});

            copyShmemToLocalUb(tensorUb, tensorTileSymmSrc, remoteId, eventIds[bufferIndex]);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

            auto tensorTileSymmDst = GetTile(
                paddingTensorDst,
                offset,
                tla::MakeShape(actualTilesNum, tileLen)
            );

            copyLocalUbToShmem(tensorTileSymmDst,
                                tensorUb,
                                rankId,
                                eventIds[bufferIndex]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

            bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
    } 

private:
    // Params params;
    static const uint32_t BUFFER_NUM = 2;
    uint32_t tmpBufferSize = 96 * 1024; // 96KB
    uint32_t COMPUTE_LENGTH = 96 * 1024 / sizeof(Element); // 48K

    AscendC::GlobalTensor<Element> gmA;
    AscendC::GlobalTensor<Element> gmWA;
    AscendC::LocalTensor<Element> tmpBuffer[BUFFER_NUM];
    AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{ 0 };
};

} // namespace Catlass::Distributed::Block

#endif  // CATLASS_DISTRIBUTE_BLOCK_BLOCK_ALLGATHER_HPP