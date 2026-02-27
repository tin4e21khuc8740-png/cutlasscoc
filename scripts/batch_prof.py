import pandas as pd
import subprocess
import sys
import os
import re

SPECIAL = 43  # 需要 group_size 的 OP_ID

def run(times, OP_ID):
    # 执行 gen_params.py 生成测试 case
    command = "python3 scripts/gen_params.py {} {}".format(times, OP_ID)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    data = pd.read_csv('./params/MNK_data.csv')
    prof_data_path = "./result/prof_data/batch_prof_data.csv"

    command = "mkdir -p result & mkdir -p result/prof_data"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    columns_default = ["M", "N", "K", "rank_size", "transB", "oper_time_us", "oper_range", "lcoc_time_us", "lcoc_range", "lccl_time", "matmul_time:", "lccl_total_time_us:", "lccl_range"]
    columns_special = ["M", "N", "K", "group_size", "rank_size", "transB", "oper_time_us", "oper_range", "lcoc_time_us", "lcoc_range", "hccl_time", "matmul_time", "hccl_total_time_us"]

    columns = columns_special if OP_ID == SPECIAL else columns_default

    for index, row in data.iterrows():
        M_val = row.iloc[0]
        N_val = row.iloc[1]
        K_val = row.iloc[2]
        if OP_ID == 43:
            group_size_val = row.iloc[3]
            ranksize_val = row.iloc[4]
            transA = row.iloc[5]
            transB = row.iloc[6]
        else:
            ranksize_val = row.iloc[3]
            transA = row.iloc[4]
            transB = row.iloc[5]

        if OP_ID == SPECIAL:
            if group_size_val is None:
                raise ValueError(f"group_size must be provided for OP_ID={SPECIAL}")
            command = f"./run_for_batch.sh {ranksize_val} {OP_ID} {M_val} {N_val} {K_val} {transB} prof {group_size_val}"
        else:
            command = f"./run_for_batch.sh {ranksize_val} {OP_ID} {M_val} {N_val} {K_val} {transB} prof"

        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output_lines = result.stdout.strip().splitlines()
        last_line = output_lines[-1]
        line = output_lines[-2]
        # print(last_line)

        match = re.search(r'oper_time:([\d.]+)', last_line)
        if match:
            oper_time = float(re.search(r'oper_time:([\d.]+)', last_line).group(1))
            lcoc_time = float(re.search(r'lcoc_time:([\d.]+)', last_line).group(1))
        else:
            print(f"[ERROR]匹配失败!输出中不存在oper_time!")
            print(last_line)
            sys.exit(1)
        # lccl_time = float(re.search(r'lccl_time:([\d.]+)', line).group(1))
        # mm_time = float(re.search(r'matmul_time:([\d.]+)', line).group(1))
        # mmlccl_time = float(re.search(r'lccl_total_time_us:([\d.]+)', line).group(1))
        # lccl_time = 0
        # mm_time = 0
        # mmlccl_time = 0

        # 提取数组
        oper_range = [float(x) for x in re.search(r'oper_range:\[([\d., ]+)\]', last_line).group(1).split(', ')]
        lcoc_range = [float(x) for x in re.search(r'lcoc_range:\[([\d., ]+)\]', last_line).group(1).split(', ')]
        # lccl_range = [float(x) for x in re.search(r'lccl_range:\[([\d., ]+)\]', line).group(1).split(', ')]
        # lccl_range = []

        if OP_ID == SPECIAL:
            # OPID=27, 解析 GMmatmul_hccl_prof.sh 的输出
            hccl_time = float(re.search(r'hccl_time:([\d.]+)', line).group(1))
            mm_time = float(re.search(r'aclnn_time:([\d.]+)', line).group(1))
            mmhccl_time = float(re.search(r'total_time:([\d.]+)', line).group(1))
            hccl_range = []
        else:
            # 其他 OPID, 解析 matmul_lccl_prof.sh 的输出
            lccl_time = float(re.search(r'lccl_time:([\d.]+)', line).group(1))
            mm_time = float(re.search(r'matmul_time:([\d.]+)', line).group(1))
            mmlccl_time = float(re.search(r'lccl_total_time_us:([\d.]+)', line).group(1))
            lccl_range = [float(x) for x in re.search(r'lccl_range:\[([\d., ]+)\]', line).group(1).split(', ')]
                
        if lcoc_time is None or oper_time is None:
            print(f"[Warning] Could not parse times for M:{M_val} N:{N_val} K:{K_val}")
            continue

        # 保存到 DataFrame
        if OP_ID == SPECIAL:
            frame = pd.DataFrame([[M_val, N_val, K_val, group_size_val, ranksize_val, transB, oper_time, oper_range, lcoc_time, lcoc_range, hccl_time, mm_time, mmhccl_time]], columns=columns)
        else:
            frame = pd.DataFrame([[M_val, N_val, K_val, ranksize_val, transB, oper_time, oper_range, lcoc_time, lcoc_range, lccl_time, mm_time, mmlccl_time, lccl_range]], columns=columns)
        
        if index == 0:
            frame.to_csv(prof_data_path, mode='w', header=True, index=False)
        else:
            frame.to_csv(prof_data_path, mode='a', header=False, index=False)
        print(last_line)

if __name__ == "__main__":
    times = int(sys.argv[1])
    OP_ID = int(sys.argv[2])

    run(times, OP_ID)
