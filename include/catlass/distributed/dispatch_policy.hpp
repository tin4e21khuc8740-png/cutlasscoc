#ifndef CATLASS_DISTRIBUTED_DISPATCH_POLICY_HPP
#define CATLASS_DISTRIBUTED_DISPATCH_POLICY_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

namespace Catlass::Distributed {
    struct CommAtlasA2AllGather {
        using ArchTag = Arch::AtlasA2;
    };

    struct CommAtlasA2ReduceScatter {
        using ArchTag = Arch::AtlasA2;
    };

    struct CommAtlasA2AllReduce {
        using ArchTag = Arch::AtlasA2;
    };

    struct CommAtlasA2All2Allv {
        using ArchTag = Arch::AtlasA2;
    };
}

#endif // CATLASS_DISTRIBUTED_DISPATCH_POLICY_HPP