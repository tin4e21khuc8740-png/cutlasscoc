import csv
import sys
import os

# 参数说明：
# 前面是一个或多个文件路径
# 最后 4 个参数分别是 M, N, K, group_size
file_paths = sys.argv[1:-4]
M = int(sys.argv[-4])
N = int(sys.argv[-3])
K = int(sys.argv[-2])
group_size = int(sys.argv[-1])

print("File Paths:", file_paths)
print(f"M={M}, N={N}, K={K}, group_size={group_size}")

results = []
all_last_five_rows = []
for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"[{file_path}] File not found.")
        continue

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        time_list = [float(row['Task Duration(us)']) for row in reader if 'Task Duration(us)' in row]
        if time_list:
            time_us = sum(time_list[-5 : ]) / 5
            Mflops = 2.0 * M * N * K * group_size * 1e-6  # 如果 group_size 会影响计算，可以乘上
            Tflops = Mflops / time_us
            results.append((file_path, time_us, Tflops))
            all_last_five_rows.append(time_list[-5 : ])
            print(f"[{file_path}] M:{M} N:{N} K:{K} group_size:{group_size} time_us:{time_us} Tflops:{Tflops}")
        else:
            print(f"[{file_path}] No valid 'Task Duration(us)' found.")

# 取最大时间对应的性能
range_value = [round((max(elements) - min(elements)), 4) for elements in zip(*all_last_five_rows)]
if results:
    maxtime = max(results, key=lambda x: x[1])  # 时间最大值
    print(f"M:{M} N:{N} K:{K} group_size:{group_size} time_us:{maxtime[1]} lccl_range:{range_value}")
