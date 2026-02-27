#!/bin/bash

# 加载 Ascend 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 检查参数数量
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <M> <N> <K> <deviceId> <mode>"
    exit 1
fi

# 获取命令行参数
M=$1
K=$2
N=$3
transA=$4
transB=$5
deviceId=$6
mode=$7  # 添加模式参数，例如 prof、batchprof 等

# 如果模式不是 prof，则生成测试数据
# if [[ $mode != "prof" ]]; then
#     python3 ./scripts/gen_golden.py $M $N $K
# fi

# 编译代码

if [[ $mode == "error" ]]; then
    cd build
    ./bin/aclmatmul_test $M $K $N $transA $transB $deviceId $mode 
fi

# 设置性能测试的输出目录
output_dir="./prof"
mkdir -p $output_dir

# 根据模式执行不同的测试
if [[ $mode == "prof" ]]; then
    rm -rf $output_dir/*
    # 使用 msprof 工具运行 aclmatmul_test 并记录性能数据
    # msprof op --warm-up=10 --application="./build/bin/aclmatmul_test $M $K $N $transA $transB $deviceId $mode" --output=$output_dir
    msprof --application="./build/bin/aclmatmul_test $M $K $N $transA $transB $deviceId $mode" --output=$output_dir
    # 分析并处理性能数据 (例如，使用 prof.py 解析)
    # python3 ./prof.py `find $output_dir -name "op_summary_*.csv" | tail -n 1` $M $N $K
    # python3 ./prof.py `find $output_dir -name "OpBasic*.csv" | tail -n 1` $M $K $N
    python3 ./prof.py `find $output_dir -name "op_summary_*.csv" | tail -n 1` $M $K $N
fi

cd ..
