M=$1
N=$2
K=$3
transB=$4
start_id=$5

cd aclnnMatmul
./run.sh $M $K $N 0 $transB $start_id prof
cd ..