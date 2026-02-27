#ifndef CATLASS_DISTRIBUTE_TILE_COPY_LOCAL_UB_TO_SHMEM_HPP
#define CATLASS_DISTRIBUTE_TILE_COPY_LOCAL_UB_TO_SHMEM_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "tla/tensor.hpp"

#include "shmem_api.h"

namespace Catlass::Distributed::Tile {

template <
    class ArchTag,
    class TensorOut_,
    class TensorIn_
>
class CopyLocalUbToShmem {
public:
    using TensorOut = TensorOut_;
    using TensorIn = TensorIn_;
    // using TensorWorkspace = TensorWorkspace_;

    CATLASS_DEVICE
    CopyLocalUbToShmem() {};

    template <class TensorOut, class TensorIn>
    CATLASS_DEVICE
    void operator() (
        TensorOut &tensorOut,
        TensorIn &tensorIn,
        uint32_t rankId,
        AscendC::TEventID eventId)
    {
        non_contiguous_copy_param copyParams;
        copyParams.repeat = tla::get<0>(tensorIn.shape());
        copyParams.length = tla::get<1>(tensorIn.shape());
        copyParams.src_ld = tla::get<0>(tensorIn.stride()); //头到头的距离
        // copyParams.dst_ld = tla::get<1, 1>(tensorOut.stride());
        copyParams.dst_ld = tla::get<0>(tensorOut.stride());

        auto dstOffset = tensorOut.layout()(tensorOut.coord());
        auto srcOffset = tensorIn.layout()(tensorIn.coord());

        shmem_mte_put_mem_nbi(tensorOut.data()[dstOffset], 
                              tensorIn.data()[srcOffset], 
                              copyParams, 
                              rankId, 
                              eventId);
    }
};

} // namespace Catlass::Distributed::Tile

#endif  // CATLASS_DISTRIBUTE_TILE_COPY_LOCAL_UB_TO_SHMEM_HPP