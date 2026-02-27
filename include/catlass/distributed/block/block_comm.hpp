#ifndef CATLASS_GEMM_BLOCK_BLOCK_COMM_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_COMM_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Distributed::Block {
template <
    class DispatchPolicy, 
    class TensorIn,
    class TensorOut
>
struct BlockComm {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockComm is not implemented for this DispatchPolicy");
};

} // namespace Catlass::Distributed::Block

#include "catlass/distributed/block/block_comm_allgather.hpp"
#include "catlass/distributed/block/block_comm_allreduce.hpp"
#include "catlass/distributed/block/block_comm_reducescatter.hpp"
// #include "catlass/distributed/block/block_comm_all2allv.hpp"
#endif // CATLASS_DISTRIBUTED_BLOCK_BLOCK_COMM_HPP