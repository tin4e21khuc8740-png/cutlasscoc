rank_size=$1
matmul_type=$2
M=$3
N=$4
K=$5
transB=$6
start_id=$7

case "$matmul_type" in
    0)
        OPNAME="00_test_lccl_allgather"
        ;;
    1)
        OPNAME="01_test_lccl_allreduce"
        ;;
    2)
        OPNAME="02_test_lccl_reducescatter"
        ;;
esac

python3 matmul_lccl_prof.py $rank_size $OPNAME $M $N $K $transB $start_id