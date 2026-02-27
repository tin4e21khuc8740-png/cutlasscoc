import pandas as pd
import subprocess
import sys
import os
import re

def run(times, OP_ID):
    # 执行gen_params.py生成测试case
    command = "python3 scripts/gen_params.py {} {}".format(times, OP_ID)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    data = pd.read_csv('./params/MNK_data.csv')
    error_data_path = "./result/error_data/batch_error_data.csv"

    command = "mkdir -p result & mkdir -p result/error_data"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if OP_ID == 43:
        columns = ["M", "N", "K", "group_size", "transA", "transB", "errCounts"]
    else:
        columns = ["M", "N", "K", "transA", "transB", "errCounts"]

    # # 设置表头
    # results = pd.DataFrame(columns=["M", "N", "K", "transA", "transB", "errCount"])
    # 测试结果保存路径
    for index, row in data.iterrows():
        M = row.iloc[0]
        N = row.iloc[1]
        K = row.iloc[2]
        if OP_ID == 43:
            group_size = row.iloc[3]
            rank_size = row.iloc[4]
            transA = row.iloc[5]
            transB = row.iloc[6]
        else:
            rank_size = row.iloc[3]
            transA = row.iloc[4]
            transB = row.iloc[5]

        if OP_ID == 43:
            command = f"./run_for_batch.sh {rank_size} {OP_ID} {M} {N} {K} {transB} error {group_size}"
        else:
            command = f"./run_for_batch.sh {rank_size} {OP_ID} {M} {N} {K} {transB} error"
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output_lines = result.stdout.strip().splitlines()

        errCounts = [-1.0]
        if OP_ID == 43:
            full_output = result.stdout    
            errCounts = [float(x) for x in re.findall(r'errorCount: (\d+)', full_output)]
            print(f"M={M}, N={N}, K={K}, group_size={group_size}, errCounts={errCounts}")

        else:
            last_line = output_lines[-1]
            print(last_line)
            errCounts = [float(x) for x in re.search(r'errorCounts:\[([\d., ]+)\]', last_line).group(1).split(', ')]

        # frame =  pd.DataFrame([[M, N, K, transA, transB, errCount]], 
        #                       columns=["M", "N", "K", "transA", "transB", "errCount"])
        if OP_ID == 43:
            frame =  pd.DataFrame([[M, N, K, group_size, transA, transB, errCounts]], 
                              columns=columns)
        else:
            frame =  pd.DataFrame([[M, N, K, transA, transB, errCounts]], 
                              columns=columns)
            
        # 保存测试结果
        if index == 0:
            frame.to_csv(error_data_path, mode='w', header=True, index=False)
        else:
            frame.to_csv(error_data_path, mode='a', header=not os.path.exists(error_data_path), index=False)
    
if __name__ == "__main__":
    times = int(sys.argv[1])
    OP_ID = int(sys.argv[2])

    run(times, OP_ID)