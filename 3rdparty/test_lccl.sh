rank_size=$1
matmul_type=$2
type=$3
M=$4
N=$5
K=$6
transB=$7
start_id=$8

output_dir="./prof"

cd synopic-baseline
rm -rf $output_dir
mkdir -p $output_dir
run_cmd="./build/bin/$matmul_type $M $N $K $transB 1 10"
mpirun -np $rank_size msprof --application="$run_cmd" --output=$output_dir

python3 ../prof.py `find $output_dir -name "op_summary_*.csv"` $M $N $K 1

# python3 examples/scripts/test_prof.py $type --rank_size=$rank_size --problem_shape=$M,$N,$K --warmup_times=5 --execute_times=5
cd ..