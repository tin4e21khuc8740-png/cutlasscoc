#!/bin/bash
# ./run.sh 16 16 16 prof 8 4
# 加载 Ascend 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 检查参数数量
if [ "$#" -ne 8 ]; then
    echo "Usage: $0 <M> <N> <K> <mode> <ranSize> <groupSize> <transB> <start_id>"
    exit 1
fi

# 获取命令行参数
M=$1
N=$2
K=$3
mode=$4  # 添加模式参数
ranksize=$5
group_size=$6
transB=$7
start_id=$8

# 如果模式不是 prof，则生成测试数据
# if [[ $mode != "prof" ]]; then
#     python3 ./scripts/gen_golden.py $M $N $K
# fi

# 编译代码

if [[ $mode == "error" ]]; then
    cd build
    for ((rank=0; rank<$ranksize; rank++)); do
        ./bin/aclgroupedmatmul_test $M $N $K $rank $mode $ranksize $group_size $transB $start_id &
    done
    wait
fi


# 根据模式执行不同的测试
if [[ $mode == "prof" ]]; then
    # 设置性能测试的输出目录
    output_dir="./prof"
    mkdir -p $output_dir
    rm -rf $output_dir/*
    # 使用 msprof 工具运行 aclmatmul_test 并记录性能数据
    for ((rank=0; rank<$ranksize; rank++)); do
        # msprof --application="./build/bin/aclgroupedmatmul_test $M $N $K $rank $mode $ranksize $group_size $transB $start_id" --output=$output_dir/rank_$rank &
        msprof op --warm-up=10 --application="./build/bin/aclgroupedmatmul_test $M $N $K $rank $mode $ranksize $group_size $transB $start_id" --output=$output_dir/rank_$rank &
    done
    wait
    # 分析并处理性能数据 (例如，使用 prof.py 解析)
    # python3 ./prof.py `find $output_dir -name "op_summary_*.csv" | tail -n 1` $M $N $K
    # python3 ./prof.py `find $output_dir -name "OpBasic*.csv" | tail -n 1` $M $K $N

    # csv_files_baseline=$(find "$output_dir" -name "op_summary_*.csv" | sort)
    # python3 ./scripts/prof.py $csv_files_baseline $M $K $N $ranksize $group_size
    python3 ./scripts/prof.py `find $output_dir -name "OpBasic*.csv" | sort` $M $K $N $ranksize $group_size
fi

cd ..

