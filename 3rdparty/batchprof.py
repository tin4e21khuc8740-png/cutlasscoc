import os
import csv
import subprocess
import sys
import re
kernel_name_in = "aclnn_lcoc"
kernel_name_out = "aclnn_lcoc"
device_id = 0
start_idx = 0

def run_profiling(command):
    for _ in range(3):  # Retry up to 3 times
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output_t = result.stdout.strip().split("lccl_time:")

        if len(output_t) < 2:
            # Handle the case where the expected output is not present
            print(f"Error: Unexpected output format - retry")
        else:
            lccl_time = float(re.findall(r"[-+]?\d*\.\d+|\d+", output_t[1])[0])

        output_tt = result.stdout.strip().split("matmul_time:")

        if len(output_tt) < 2:
            # Handle the case where the expected output is not present
            print(f"Error: Unexpected output format - retry")
        else:
            matmul_time = float(re.findall(r"[-+]?\d*\.\d+|\d+", output_tt[1])[0])

        return lccl_time, matmul_time

    # Return -1 if unable to get valid actlassTime after 3 retries
    print(f"Error: Unable to retrieve valid actlassTime for command - {command}")
    return -1

def batch_prof(kernel_name_in, kernel_name_out, rank_size, matmul_type, device_id):
    
    # params_filepath = kernel_name_in + "_data.csv"
    params_filepath = "/home/workspace/cdx/luluteam_coc/params/MNK_data.csv"
    accur_output_filepath = kernel_name_out + "_matmul_type" + str(matmul_type) + "_rank_size" + str(rank_size) + "_prof.csv"

    fieldnames = [
        "M",
        "N",
        "K",
        "rank_size",
        "lccl_time(us)",
        "matmul_time(us)",
        "lccl_total_time(us)"
    ]

    with open(accur_output_filepath, "a+", newline="") as output_csvfile:
        writer = csv.writer(output_csvfile)
        writer.writerow(fieldnames)

    with open(params_filepath) as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            if index < start_idx:  # Skip header row
                continue

            M = int(row["M"])
            N = int(row["N"])
            K = int(row["K"])

            command = f"bash ./matmul_lccl_prof.sh {rank_size} {matmul_type} {M} {N} {K}"

            lccl_time, matmul_time = run_profiling(command)

            total_time = lccl_time + matmul_time

            # Tflops = 2 * M * N * K * 1e-9 / actlassTime
            with open(accur_output_filepath, "a+", newline="") as output_csvfile:
                writer = csv.DictWriter(output_csvfile, fieldnames)
                writer.writerow({
                    "M": M,
                    "N": N,
                    "K": K,
                    "rank_size": rank_size,
                    "lccl_time(us)": lccl_time,
                    "matmul_time(us)": matmul_time,
                    "lccl_total_time(us)": total_time
                })

            print(M, N, K, rank_size, lccl_time, matmul_time, total_time)

if __name__ == "__main__":

    rank_size=int(sys.argv[1])
    matmul_type=int(sys.argv[2])

    batch_prof(kernel_name_in, kernel_name_out, rank_size, matmul_type, device_id)

