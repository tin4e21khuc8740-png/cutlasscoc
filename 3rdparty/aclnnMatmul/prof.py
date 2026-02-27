import csv
import sys
import os

file_path = sys.argv[1]
print("File Path:", file_path)
M = int(sys.argv[2])
K = int(sys.argv[3])
N = int(sys.argv[4])

# 从 CSV 文件中读取任务持续时间
with open(file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    time_list = [float(row['Task Duration(us)']) for row in reader if 'Task Duration(us)' in row]
    # time_us = sum(time_list[5:]) / 5;
    time_us = time_list[0];

# 如果找到任务持续时间，则计算 FLOPS
if time_us is not None:
    Mflops = 2.0 * M * N * K * 1e-6
    Tflops = Mflops / time_us
    utilization_ratio = Tflops / 294.91

    print("M:", M, "K:", K, "N:", N, "time_us:", time_us, "Tflops:", Tflops, "utilization_ratio:", utilization_ratio)

    # # 输出结果到 CSV 文件
    # output_dir = './result/prof_data'
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, 'acl_gemm_prof_info.csv')

    # # 写入 CSV 文件
    # with open(output_file, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["M", "N", "K", "time_us", "Tflops", "utilization_ratio"])
    #     writer.writerow([M, N, K, time_us, Tflops, utilization_ratio])

    # print(f"Results written to {output_file}")
else:
    print("No valid 'Task Duration(us)' found in the file.")
