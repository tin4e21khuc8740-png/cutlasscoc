rank_size=$1
M=$2
N=$3
K=$4
group_size=$5
transB=$6
start_id=$7

python3 GMmatmul_hccl_prof.py $rank_size $M $N $K $group_size $transB $start_id