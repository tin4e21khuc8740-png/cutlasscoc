# 检查参数数量
if [ "$#" -lt 7 ] || [ "$#" -gt 8 ]; then
    echo "Usage: <rank_size> <OPID> <M> <N> <K> <transB> <mode> <group_size> "
    exit 1
fi

rank_size=$1
OPID=$2
M=$3
N=$4
K=$5
transB=$6
mode=$7
group_size=${8:-1}

#卡号和端口号都不能冲突
start_id=0
IPPORT="tcp://127.0.0.1:8777"

if [[ $((start_id + ${rank_size})) -gt 16 ]]; then
    echo "run.sh: rankId不能大于16"
    echo "当前 start_id=${start_id}, rank_size=${rank_size}"
    exit 1
fi

case "$OPID" in
    40)
        OPNAME="40_allgather_matmul"
        OPNAME_baseline="00_test_lcoc_allgather_matmul"
        OP_ID=0
        ;;
    41)
        OPNAME="41_matmul_allreduce"
        OPNAME_baseline="01_test_lcoc_matmul_allreduce"
        OP_ID=1
        ;;
    42)
        OPNAME="42_matmul_reducescatter"
        OPNAME_baseline="02_test_lcoc_matmul_reducescatter"
        OP_ID=2
        ;;
    43)
        OPNAME="43_grouped_matmul_alltoallv"
        OPNAME_baseline="03_test_lcoc_grouped_matmul_alltoallv"
        OP_ID=3
        ;;
esac

if [[ $mode == "error" ]]; then
    mkdir -p ./examples/${OPNAME}/data
    rm -rf ./examples/${OPNAME}/data/*

    wait
    mpirun -np $rank_size ./output/bin/${OPNAME} "$M" "$N" "$K" "$IPPORT" 10 0
    wait

    if [[ "$OPID" -eq 41 ]]; then
        python3 ./examples/${OPNAME}/base.py $M $N $K $rank_size $transB $start_id
    fi

    if [[ "$OPID" -eq 42 ]]; then
        python3 ./examples/${OPNAME}/base.py $M $N $K $rank_size $transB $start_id
    fi

    if [[ "$OPID" -eq 43 ]]; then
        python3 ./examples/${OPNAME}/base.py $M $N $K $rank_size $transB $start_id $group_size
    fi
    # 验证计算结果
    python3 scripts/compare.py ${OPNAME} $M $N $K $rank_size $transB $start_id $group_size
    wait
fi

if [[ $mode == "prof" ]]; then
    # 创建算子和两个基线的文件夹
    output_dir_baseline_lcoc="./prof_baseline/lcoc"
    output_dir_baseline_lccl="./prof_baseline/lccl"
    output_dir_us="./prof_us"

    mkdir -p "$output_dir_baseline_lcoc" "$output_dir_baseline_lccl" "$output_dir_us" 
    rm -rf "$output_dir_baseline_lcoc"/* "$output_dir_baseline_lccl"/* "$output_dir_us"/* 

    # baseline profiling
    if [[ "$OPID" -ne 43 ]]; then
        run_cmd="./3rdparty/synopic-baseline/build/bin/$OPNAME_baseline $M $N $K $transB 1 10"
    else
        run_cmd="./3rdparty/synopic-baseline/build/bin/$OPNAME_baseline $M $N $K $transB $group_size 1 1 10"
    fi
    mpirun -np $rank_size msprof --application="$run_cmd" --output=$output_dir_baseline_lcoc
    wait

    # 执行算子记录性能数据
    mpirun -np $rank_size msprof --application="./output/bin/${OPNAME} $M $N $K $IPPORT 10 1" --output="$output_dir_us" 
    wait 

    # 查找基线、算子的所有op文件路径
    csv_files_baseline=$(find "$output_dir_baseline_lcoc" -name "op_summary_*.csv" | sort)
    csv_files_us=$(find "$output_dir_us" -name "op_summary_*.csv" | sort)

    # 合并成一个空格分隔的列表
    csv_files_all="$csv_files_baseline $csv_files_us"
    
    # 终端输出lccl + mm的性能数据
    cd 3rdparty
    if [[ "$OPID" -eq 43 ]]; then
        # OPID=27, 运行 GMmatmul_hccl_prof.sh
        ./GMmatmul_hccl_prof.sh $rank_size $M $N $K $group_size $transB $start_id
    else
        ./matmul_lccl_prof.sh $rank_size $OP_ID $M $N $K $transB $start_id
    fi
    cd ..
    wait

    # 输出lcoc和oper的性能数据
    if [[ "$OPID" -ne 43 ]]; then
        python3 prof.py $OPID $rank_size $M $N $K $transB $csv_files_all 
    else
        python3 prof.py $OPID $rank_size $M $N $K $group_size $transB $csv_files_all 
    fi
    wait
fi