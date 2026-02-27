import pandas as pd
import subprocess
import re
import sys
import os

HCCL_SCRIPT = "./run_hccl.sh"
ACLNN_SCRIPT = "./test_aclnnGM.sh"

def parse_aclnn_output(output_lines):
    time_aclnn_val = None
    for line in reversed(output_lines):
        if "time_us:" in line:
            match = re.search(r'time_us: ([\d.]+)', line)
            if match:
                time_aclnn_val = float(match.group(1))
                break
    return time_aclnn_val

def parse_hccl_output(output_text):
    pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\w+)\s*$", 
        re.MULTILINE
    )
    match = pattern.search(output_text)
    
    if match:
        # group(2) 是 aveg_time_us
        return float(match.group(2))
    else:
        return None

def run(rank_size, m, n, k, group_size, transB, start_id):
    
    hccl_time_us = None
    aclnn_time_us = None

    if not os.path.isfile(HCCL_SCRIPT) or not os.path.isfile(ACLNN_SCRIPT):
        print(f"[ERROR] 脚本文件 ({HCCL_SCRIPT} 或 {ACLNN_SCRIPT}) 不存在。")
        sys.exit(1)

    print("\n--- 1. HCCL ---")
    
    # 构建 HCCL 命令 (./run_hccl.sh <RANK_SIZE> 7 <M> <N> <K>)
    command_hccl = f"{HCCL_SCRIPT} {rank_size} 7 {m} {n} {k}"
    print(f"[CMD] {command_hccl}")
    
    try:
        result_hccl = subprocess.run(
            command_hccl,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        
        hccl_time_us = parse_hccl_output(result_hccl.stdout)
        
        if hccl_time_us is not None:
            print(f"[SUCCESS] HCCL time: {hccl_time_us} us")
        else:
            print("[ERROR] HCCL 命令执行成功，但无法解析时间。")
            print(f"STDOUT:\n{result_hccl.stdout}")
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] HCCL (test_hccl.sh) 命令执行失败。")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")

    print("\n--- 2. ACLNN ---")
    
    # 构建 ACLNN 命令 (./test_aclnnGM.sh <M> <N> <K> <ranksize> <groupSize>)
    command_aclnn = f"{ACLNN_SCRIPT} {m} {n} {k} {rank_size} {group_size} {transB} {start_id}"
    print(f"[CMD] {command_aclnn}")
    
    try:
        result_aclnn = subprocess.run(
            command_aclnn,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        output_lines = result_aclnn.stdout.strip().splitlines()
        
        aclnn_time_us = parse_aclnn_output(output_lines)

        if aclnn_time_us is not None:
            print(f"[SUCCESS] ACLNN time: {aclnn_time_us} us")
        else:
            print(f"[Warning] 找不到 'time_us:'。")
            print(f"Full STDOUT:\n{result_aclnn.stdout}")
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ACLNN (test_aclnnGM.sh) 命令执行失败。")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
    
    # 准备用于打印的字符串，处理 N/A 的情况
    hccl_str = f"{hccl_time_us:.2f}" if hccl_time_us is not None else "N/A"
    aclnn_str = f"{aclnn_time_us:.2f}" if aclnn_time_us is not None else "N/A"
    
    total_str = "N/A"
    if hccl_time_us is not None and aclnn_time_us is not None:
        total_time = hccl_time_us + aclnn_time_us
        total_str = f"{total_time:.2f}"
    
    print("\nType:GroupedMatmulalltoallv")
    print(f"M:{m} N:{n} K:{k} group_size:{group_size} rankSize:{rank_size} hccl_time:{hccl_str} aclnn_time:{aclnn_str} total_time:{total_str}")

if __name__ == "__main__":
    try:
        rank_size = int(sys.argv[1])
        m = int(sys.argv[2])
        n = int(sys.argv[3])
        k = int(sys.argv[4])
        group_size = int(sys.argv[5])
        transB = int(sys.argv[6])
        start_id=int(sys.argv[7])
    except ValueError as e:
        print(f"[ERROR] 参数类型错误，请确保 M,N,K,rank_size,group_size 均为数字。")
        print(f"错误详情: {e}")
        sys.exit(1)

    # 调用主函数
    run(rank_size, m, n, k, group_size, transB, start_id)