import pandas as pd
import subprocess
import sys
import os
import re

<<<<<<< Updated upstream
SPECIAL = 27  # 需要 group_size 的 OP_ID

HCCL_SCRIPT_DIR = '../template-library-phase-ii/examples/scripts/'
HCCL_RUNNER = os.path.join(HCCL_SCRIPT_DIR, 'test_prof.py')
# HCCL 的固定参数
HCCL_OP_ID = 7
HCCL_WARMUP_TIMES = 15
HCCL_EXECUTE_TIMES = 5
HCCL_SKIP_BUILD = True
# HCCL_SKIP_BUILD = False

# 解析 HCCL (test_prof.py) 的输出
def parse_hccl_output(output_text):
    pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\w+)\s*$", 
        re.MULTILINE
    )
    
    match = pattern.search(output_text)
    
    if match:
        return {
            "data_size": int(match.group(1)),
            "aveg_time_us": float(match.group(2)),
            "alg_bandwidth_gbs": float(match.group(3)),
            "status": match.group(4)
        }
    else:
        return None

def run(times, rank_size, OP_ID, mode):
    # 执行 gen_params.py 生成测试 case
    # subprocess.run(f"python3 ./scripts/gen_params.py {times} {mode}", shell=True, check=True)
=======
# 解析 aclnn (run.sh) 的输出
def parse_aclnn_output(output_lines):
    time_us_line = ""
    time_aclnn_val = None

    for line in reversed(output_lines):
        if "time_us:" in line:
            time_us_line = line.strip()
            match = re.search(r'time_us: ([\d.]+)', time_us_line)
            if match:
                time_aclnn_val = float(match.group(1))
                break # 找到后退出循环
    
    return time_aclnn_val, time_us_line

def run(times, rank_size, mode):
    
    # 执行 gen_params.py 生成测试 case
    subprocess.run(f"python3 ./scripts/gen_params.py {times} {mode}", shell=True, check=True)
>>>>>>> Stashed changes

    data = pd.read_csv('./params/MNK_data.csv')
    prof_data_path = "./result/prof_data/batch_prof_data.csv"

    columns = [
        "M", "N", "K", "group_size", 
<<<<<<< Updated upstream
        "time_aclnn_us", "time_hccl_us", "time_total_us", "hccl_status"
    ]
=======
        "time_aclnn_us"
    ]
    
    # 在循环开始前写入表头
    try:
        os.makedirs(os.path.dirname(prof_data_path), exist_ok=True)
        pd.DataFrame(columns=columns).to_csv(prof_data_path, mode='w', header=True, index=False)
    except OSError as e:
        print(f"[ERROR] 创建结果文件或目录失败: {e}")
        sys.exit(1)

>>>>>>> Stashed changes

    for index, row in data.iterrows():
        M_val = row.iloc[0]
        N_val = row.iloc[1]
        K_val = row.iloc[2]
        group_size_val = row.iloc[3]

        problem_shape = f"{M_val},{N_val},{K_val}"
        print(f"\n--- Case {index + 1}/{len(data)}: {problem_shape}, group={group_size_val} ---")

        print(f"[INFO] 正在运行 Test aclnn...")

<<<<<<< Updated upstream
        if OP_ID == SPECIAL:
            if group_size_val is None:
                raise ValueError(f"group_size must be provided for OP_ID={SPECIAL}")
            command = f"./run.sh {M_val} {N_val} {K_val} prof {rank_size} {group_size_val}"
            print(f"\n[INFO] Running command: {command}")
        else:
            command = f"./run.sh {rank_size} {OP_ID} {M_val} {N_val} {K_val} prof"

        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        output_lines = result.stdout.strip().splitlines()
        time_us_line = ""

        time_aclnn_val = None

        # 从后向前查找最后一条匹配的 "time_us" 行
        for line in reversed(output_lines):
            if "time_us:" in line:
                time_us_line = line.strip()
                match = re.search(r'time_us: ([\d.]+)', time_us_line)
                if match:
                    time_aclnn_val = float(match.group(1))
                    break # 找到后退出循环
        
        # 打印找到的行
        if time_us_line:
            print(f"Captured output line: {time_us_line}")
        else:
            print(f"[Warning] Could not find 'time_us:' in output for M:{M_val} N:{N_val} K:{K_val} G:{group_size_val}")
            print(f"Full STDOUT:\n{result.stdout}") # 打印完整输出以便调试

        print(f"[INFO] 正在运行 Test HCCL...")
        time_HCCL_val = None
        hccl_status = "pending"

        command_B_list = [
            "python3", "test_prof.py", str(HCCL_OP_ID),
            f"--rank_size={rank_size}",
            f"--problem_shape={problem_shape}",
            f"--warmup_times={HCCL_WARMUP_TIMES}",
            f"--execute_times={HCCL_EXECUTE_TIMES}"
        ]
        if HCCL_SKIP_BUILD:
            command_B_list.append("--skip_build")
        
        command_B_str = " ".join(command_B_list)
        print(f"[CMD-B] {command_B_str}")
        print(f"[CMD-B CWD] {HCCL_SCRIPT_DIR}")

        try:
            result_B = subprocess.run(
                command_B_str, 
                shell=True, 
                capture_output=True, 
                text=True, 
                check=True, # 如果HCCL脚本失败，将引发异常
                encoding='utf-8', 
                cwd=HCCL_SCRIPT_DIR # 指定执行目录
            )
            
            parsed_data_B = parse_hccl_output(result_B.stdout)
            
            if parsed_data_B:
                time_HCCL_val = parsed_data_B['aveg_time_us']
                hccl_status = parsed_data_B['status']
                print(f"[SUCCESS-B] aveg_time: {time_HCCL_val} us, status: {hccl_status}")
            else:
                hccl_status = "parse_error"
                print(f"[ERROR-B] HCCL 命令成功，但无法解析其输出。")

        except subprocess.CalledProcessError as e:
            hccl_status = "run_failed"
            print(f"[ERROR-B] HCCL 命令执行失败。")
            print(f"STDERR:\n{e.stderr}")
        
        except FileNotFoundError:
            hccl_status = "file_not_found"
            print(f"[ERROR-B] 无法运行 HCCL 命令。路径是否正确? {HCCL_RUNNER}")

        time_total_val = None
        if time_aclnn_val is not None and time_HCCL_val is not None:
            time_total_val = time_aclnn_val + time_HCCL_val
            print(f"[INFO] 总时间 (A+B): {time_total_val} us")
        else:
            print(f"[WARN] 无法计算总时间 (A: {time_aclnn_val}, B: {time_HCCL_val})")

        # 保存到 DataFrame
        frame = pd.DataFrame([[M_val, N_val, K_val, group_size_val, time_aclnn_val, time_HCCL_val, time_total_val, hccl_status]], columns=columns)
        if index == 0:
            frame.to_csv(prof_data_path, mode='w', header=True, index=False)
        else:
            frame.to_csv(prof_data_path, mode='a', header=False, index=False)
        print(f"[SUCCESS] Saved: M={M_val}, N={N_val}, K={K_val}, group_size={group_size_val}, time_aclnn={time_aclnn_val}, time_hccl={time_HCCL_val}, time_Total={time_total_val}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
            print("Usage: python batch_prof.py <times> <rank_size> <OP_ID> <mode>")
=======
        command = f"./run.sh {M_val} {N_val} {K_val} prof {rank_size} {group_size_val}"

        print(f"[CMD] {command}")

        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                check=True,
                encoding='utf-8'
            )
            
            output_lines = result.stdout.strip().splitlines()
            
            # 解析 aclnn 输出
            time_aclnn_val, time_us_line = parse_aclnn_output(output_lines)
            
            if time_aclnn_val is not None:
                print(f"Captured output line: {time_us_line}")
            else:
                print(f"[Warning] 找不到 'time_us:'。M:{M_val} N:{N_val} K:{K_val} G:{group_size_val}")
                print(f"Full STDOUT:\n{result.stdout}")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] aclnn (run.sh) 命令执行失败。")
            print(f"STDERR:\n{e.stderr}")
            time_aclnn_val = None
        
        except FileNotFoundError:
            print(f"[ERROR] 找不到 './run.sh'。请检查文件是否存在并有执行权限。")
            sys.exit(1) # 如果 run.sh 找不到，后续也无法执行，直接退出

        # 保存到 DataFrame
        frame = pd.DataFrame(
            [[M_val, N_val, K_val, group_size_val, time_aclnn_val]], 
            columns=columns
        )
        
        frame.to_csv(prof_data_path, mode='a', header=False, index=False)
        
        print(f"[SUCCESS] Saved: M={M_val}, N={N_val}, K={K_val}, group_size={group_size_val}, time_aclnn={time_aclnn_val}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
            print("Usage: python batch_prof.py <times> <rank_size> <mode>")
>>>>>>> Stashed changes
            sys.exit(1)

    times = int(sys.argv[1])
    rank_size = int(sys.argv[2])
<<<<<<< Updated upstream
    OP_ID = int(sys.argv[3])
    mode = int(sys.argv[4])

    run(times, rank_size, OP_ID, mode)
=======
    mode = sys.argv[3]

    run(times, rank_size, mode)
>>>>>>> Stashed changes
