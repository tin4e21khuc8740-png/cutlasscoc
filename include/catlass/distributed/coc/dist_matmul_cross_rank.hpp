#ifndef CATLASS_DISTRIBUTE_COMM_ALLGATHER_MATMUL_HPP
#define CATLASS_DISTRIBUTE_COMM_ALLGATHER_MATMUL_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Distributed::Kernel {

template <
    class GemmKernel_,
    class DistSchedule_,
    class BlockComm_
>
class AllGatherMatmul {
    
};

template <
    class GemmKernel_,
    class DistSchedule_,
    class BlockComm_
>
class MatmulAllReduce {
    
};

template <
    class GemmKernel_,
    class DistSchedule_,
    class BlockComm_
>
class MatmulReduceScatter {
    
};

} // namespace Distributed::Kernel

#endif  // CATLASS_DISTRIBUTE_KERNEL_DIST_MATMUL_HPP