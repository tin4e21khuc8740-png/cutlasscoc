M=$1
N=$2
K=$3
ranksize=$4
group_size=$5
transB=$6
start_id=$7

cd aclnnGroupedMatmul
./run.sh $M $N $K prof $ranksize $group_size $transB $start_id
cd ..