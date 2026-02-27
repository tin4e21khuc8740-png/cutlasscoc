#!/bin/bash
set -e

if [ "$#" -ne 5 ]; then
    echo "[ERROR] 参数不足。"
    echo "用法: $0 <OP_ID> <RANK_SIZE> <M> <N> <K>"
    exit 1
fi

# OP_ID 为 7
RANK_SIZE=$1
OP_ID=$2
M=$3
N=$4
K=$5

WARMUP_TIMES=15
EXECUTE_TIMES=5
SCRIPT_TO_RUN="synopic-baseline/examples/scripts/test_prof.py"
PROBLEM_SHAPE="${M},${N},${K}"
SKIP_BUILD=true # 可选 'true' 或 'false'
MPI_MPICH=false # 可选 'true' 或 'false'

echo "[INFO] 运行 OP_ID=$OP_ID, RANK_SIZE=$RANK_SIZE, M=$M, N=$N, K=$K"

ARGS=(
    "$SCRIPT_TO_RUN"
    "$OP_ID"
    "--rank_size=$RANK_SIZE"
    "--problem_shape=$PROBLEM_SHAPE"
    "--warmup_times=$WARMUP_TIMES"
    "--execute_times=$EXECUTE_TIMES"
)

# 如果 SKIP_BUILD 为 true, 添加 --skip_build
if [ "$SKIP_BUILD" = true ]; then
    ARGS+=("--skip_build")
fi

# 如果 MPI_MPICH 为 true, 添加 --mpi=mpich
if [ "$MPI_MPICH" = true ]; then
    ARGS+=("--mpi=mpich")
fi

python3 "${ARGS[@]}"