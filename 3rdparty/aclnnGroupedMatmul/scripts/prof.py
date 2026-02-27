import csv
import sys
import os


args = sys.argv[1:]  # 去掉脚本名

group_size = int(args[-1])
rank_size = int(args[-2])
N = int(args[-3])
K = int(args[-4])
M = int(args[-5])
file_path = args[:-5]
print("!!!!!!!!!!!aclnn prof File Path:", file_path)



# 从 CSV 文件中读取任务持续时间
# with open(file_path, newline='') as csvfile:
#     reader = csv.DictReader(csvfile)
#     time_list = [float(row['Task Duration(us)']) for row in reader if 'Task Duration(us)' in row]
#     time_us = time_list[0]

# 遍历所有文件，取Task Duration(us)那列的后五行的平均值作为该rank的运行时间，然后找出所有rank中运行时间的最大值

rank_time_list = []

for path in file_path:
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # 取出该文件中所有 'Task Duration(us)' 的浮点值
        time_list = [float(row['Task Duration(us)']) for row in reader if 'Task Duration(us)' in row and row['Task Duration(us)']]
        if not time_list:
            continue
        # 取最后五个值（不足 5 个则取全部）
        last_five = time_list[-5:]
        avg_time = sum(last_five) / len(last_five)
        rank_time_list.append((path, avg_time))

# 找出运行时间最大的 rank
if rank_time_list:
    max_rank_path, max_rank_time = max(rank_time_list, key=lambda x: x[1])
    print(f"最大平均运行时间: {max_rank_time} 来自文件: {max_rank_path}")
else:
    print("aclnn prof.py: 未找到任何有效的 Task Duration(us) 数据。")

time_us = max_rank_time



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
