import os
import csv
import subprocess
import sys
import re

device_id = 0
def test_prof(rank_size, matmul_type, M, N, K, transB, start_id):
    if (matmul_type == "00_test_lccl_allgather"):
        command_lccl = f"bash ./test_lccl.sh {rank_size} {matmul_type} 4 {M} {N} {K} {transB}"
        command_acl = f"bash ./test_aclnn.sh {rank_size * M} {N} {K} {transB} {start_id}"

        result = subprocess.run(command_lccl, shell=True, capture_output=True, text=True)
        output = result.stdout.strip().split("time_us:")
        output=re.findall(r"[-+]?\d*\.\d+|\d+", output[1])
        lcclTime = float(output[0])

        output0 = result.stdout.strip().split("lccl_range:")

        result = subprocess.run(command_acl, shell=True, capture_output=True, text=True)
        output = result.stdout.strip().split("time_us: ")
        output=re.findall(r"[-+]?\d*\.\d+|\d+", output[1])
        aclTime = float(output[0])

        total_time = lcclTime + aclTime
        print(f"Type:allgather_matmul M:{M} N:{N} K:{K} rank_size:{rank_size} lccl_time:{lcclTime} matmul_time:{aclTime} lccl_total_time_us:{total_time} lccl_range:{output0[1]} ")
    elif (matmul_type == "01_test_lccl_allreduce"):
        command_lccl = f"bash ./test_lccl.sh {rank_size} {matmul_type} 5 {M} {N} {K} {transB}"
        command_acl = f"bash ./test_aclnn.sh {M} {N} {K} {transB} {start_id}"

        result = subprocess.run(command_acl, shell=True, capture_output=True, text=True)
        output = result.stdout.strip().split("time_us: ")
        output=re.findall(r"[-+]?\d*\.\d+|\d+", output[1])
        aclTime = float(output[0])

        result = subprocess.run(command_lccl, shell=True, capture_output=True, text=True)
        output = result.stdout.strip().split("time_us:")
        output=re.findall(r"[-+]?\d*\.\d+|\d+", output[1])
        lcclTime = float(output[0])
        
        # data = []
        # for i in range(rank_size):
        #     line = result.stdout.strip().splitlines()[-1 - i]
        #     line = line.strip().split("task time(us): ")
        #     temp = re.findall(r"[-+]?\d*\.\d+|\d+", line[1])
        #     data.append(temp[0])
        # # output=re.findall(r"[-+]?\d*\.\d+|\d+", output[1])
        # lcclTime = float(min(data))

        # output0 = float(re.search(r'time_us::([\d.]+)', result).group(1))
        # range = [float(x) for x in re.search(r'lccl_range:\[([\d., ]+)\]', result).group(1).split(', ')]

        output0 = result.stdout.strip().split("lccl_range:")
        # output0 = re.findall(r"[-+]?\d*\.\d+|\d+", output0[1])
        # range = output0

        total_time = lcclTime + aclTime
        print(f"Type:matmul_allreduce M:{M} N:{N} K:{K} rank_size:{rank_size} lccl_time:{lcclTime} matmul_time:{aclTime} lccl_total_time_us:{total_time} lccl_range:{output0[1]}")
    elif (matmul_type == "02_test_lccl_reducescatter"):
        command_lccl = f"bash ./test_lccl.sh {rank_size} {matmul_type} 6 {M} {N} {K} {transB}"
        command_acl = f"bash ./test_aclnn.sh {rank_size * M} {N} {K} {transB} {start_id}"

        result = subprocess.run(command_acl, shell=True, capture_output=True, text=True)
        output = result.stdout.strip().split("time_us: ")
        output=re.findall(r"[-+]?\d*\.\d+|\d+", output[1])
        aclTime = float(output[0])

        result = subprocess.run(command_lccl, shell=True, capture_output=True, text=True)
        output = result.stdout.strip().split("time_us:")
        output=re.findall(r"[-+]?\d*\.\d+|\d+", output[1])
        lcclTime = float(output[0])

        output0 = result.stdout.strip().split("lccl_range:")

        total_time = lcclTime + aclTime
        print(f"Type:matmul_reducescatter M:{M} N:{N} K:{K} rank_size:{rank_size} lccl_time:{lcclTime} matmul_time:{aclTime} lccl_total_time_us:{total_time} lccl_range:{output0[1]} ")



if __name__ == "__main__":
    rank_size=int(sys.argv[1])
    matmul_type=sys.argv[2]
    M=int(sys.argv[3])
    N=int(sys.argv[4])
    K=int(sys.argv[5])
    transB = int(sys.argv[6])
    start_id=int(sys.argv[7])
    
    test_prof(rank_size, matmul_type, M, N, K, transB, start_id)