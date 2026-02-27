/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_COMMON_OPTIONS_HPP
#define EXAMPLES_COMMON_OPTIONS_HPP

#include <iostream>
#include <string>

#include "catlass/gemm_coord.hpp"
#include "catlass/gemv_coord.hpp"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#ifndef CATLASS_EXAMPLE_NAME
#define CATLASS_EXAMPLE_NAME catlass_example
#endif

/**
 * @struct GemmOptions
 * @brief Options structuture for gemm examples.
 * @brief Arguments: `example_name m n k [device_id]`
 */
struct GemmOptions {
    const std::string HELPER = "m n k [device_id]";

    Catlass::GemmCoord problemShape{128, 128, 128};
    int32_t deviceId{0};

    GemmOptions() = default;

    int Parse(int argc, const char **argv) {
        enum class ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<uint32_t>(ArgsIndex::ARGS_MAX)
            || argc < static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)) {
            std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << " " << HELPER << std::endl;
            return -1;
        }

        problemShape.m() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::M_INDEX)]);
        problemShape.n() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::N_INDEX)]);
        problemShape.k() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::K_INDEX)]);
        if (argc == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }
        return 0;
    }
};

/**
 * @struct GemvOptions
 * @brief Options structuture for gemv examples.
 * @brief Arguments: `example_name m n [device_id]`
 */
struct GemvOptions {
    const std::string HELPER = "m n [device_id]";

    Catlass::GemvCoord problemShape{128, 128};
    int32_t deviceId{0};

    GemvOptions() = default;

    int Parse(int argc, const char **argv) {
        enum class ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<uint32_t>(ArgsIndex::ARGS_MAX)
            || argc < static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)) {
            std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << " " << HELPER << std::endl;
            return -1;
        }

        problemShape.m() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::M_INDEX)]);
        problemShape.n() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::N_INDEX)]);
        if (argc == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }
        return 0;
    }
};

/**
 * @struct GroupedGemmOptions
 * @brief Options structuture for grouped/batched gemm examples.
 * @brief Arguments: `example_name problem_count m n k [device_id]`
 */
struct GroupedGemmOptions {
    const std::string HELPER = "problem_count m n k [device_id]";

    Catlass::GemmCoord problemShape{128, 128, 128};
    uint32_t problemCount{1};
    int32_t deviceId{0};

    GroupedGemmOptions() = default;

    int Parse(int argc, const char **argv) {
        enum class ArgsIndex {
            GROUP_COUNT = 1,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > static_cast<uint32_t>(ArgsIndex::ARGS_MAX)
            || argc < static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)) {
            std::cerr << TOSTRING(CATLASS_EXAMPLE_NAME) << " " << HELPER << std::endl;
            return -1;
        }
        problemCount = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::GROUP_COUNT)]);
        problemShape.m() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::M_INDEX)]);
        problemShape.n() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::N_INDEX)]);
        problemShape.k() = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::K_INDEX)]);
        if (argc == static_cast<uint32_t>(ArgsIndex::ARGS_MAX)) {
            deviceId = std::atoi(argv[static_cast<uint32_t>(ArgsIndex::DEVICE_ID_INDEX)]);
        }
        return 0;
    }
};

struct CommAGOptions {
    const std::string HELPER = "40_allgather_matmul m n k rankSize";

    Catlass::GemmCoord problemShape{128, 128, 128};
    int32_t rankSize;
    uint32_t rankId;
    const char *ipport;
    uint32_t repeat_time;
    bool enableProf;
    
    CommAGOptions() = default;

    int Parse(int argc, char **argv, int32_t rankIdValue, int32_t rankSizeValue)
    {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            IPPORT_INDEX,
            REPEAT_NUM,
            ENABLEPROF_INDEX,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= K_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        } 

        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        ipport = argv[IPPORT_INDEX];
        repeat_time = std::atoi(argv[REPEAT_NUM]);
        enableProf = std::atoi(argv[ENABLEPROF_INDEX]);
        rankId = rankIdValue;
        rankSize = rankSizeValue;

        return 0;
    }
};

struct CommAROptions {
    const std::string HELPER = "41_matmul_allreduce m n k rankSize";

    Catlass::GemmCoord problemShape{128, 128, 128};
    Catlass::MatrixCoord commBlockShape{static_cast<uint32_t>(128), static_cast<uint32_t>(256)};
    int32_t rankSize;
    uint32_t rankId;
    const char *ipport;
    uint32_t repeat_time;
    bool enableProf;
    
    CommAROptions() = default;

    int Parse(int argc, char **argv, int32_t rank, int32_t ranksize)
    {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            IPPORT_INDEX,
            REPEAT_NUM,
            ENABLEPROF_INDEX,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= K_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        } 

        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        ipport = argv[IPPORT_INDEX];
        repeat_time = std::atoi(argv[REPEAT_NUM]);
        enableProf = std::atoi(argv[ENABLEPROF_INDEX]);

        commBlockShape.column() = problemShape.n();
        commBlockShape.row() = problemShape.m();
        rankId = rank;
        rankSize = ranksize;

        return 0;
    }
};

struct CommRSOptions {
    const std::string HELPER = "25_matmul_allreduce m n k rankSize";

    Catlass::GemmCoord problemShape{128, 128, 128};
    Catlass::MatrixCoord commBlockShape{static_cast<uint32_t>(128), static_cast<uint32_t>(256)};
    int32_t rankSize;
    uint32_t rankId;
    const char *ipport;
    uint32_t repeat_time;
    
    CommRSOptions() = default;

    int Parse(int argc, char **argv, int32_t rank, int32_t ranksize)
    {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            IPPORT_INDEX,
            REPEAT_NUM,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= K_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        } 

        rankId = rank;
        rankSize = ranksize;
        problemShape.m() = std::atoi(argv[M_INDEX]) * rankSize;
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        ipport = argv[IPPORT_INDEX];
        repeat_time = std::atoi(argv[REPEAT_NUM]);

        commBlockShape.column() = problemShape.n();
        commBlockShape.row() = std::atoi(argv[M_INDEX]);
        

        return 0;
    }
};


#endif