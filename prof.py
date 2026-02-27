import csv
import sys
import os
from math import ceil

SPECIAL = 43

matmul_type = int(sys.argv[1])
rank_size = int(sys.argv[2])
M = int(sys.argv[3])
N = int(sys.argv[4])
K = int(sys.argv[5])
if matmul_type == SPECIAL:  # 带 group_size
    group_size = int(sys.argv[6])
    transB = int(sys.argv[7])
    file_paths = sys.argv[8:]
else:  # 普通类型
    transB = int(sys.argv[6])
    file_paths = sys.argv[7:]

# 将 file_paths 按 rank_size 分组
num_groups = ceil(len(file_paths) / rank_size)
groups = [file_paths[i * rank_size:(i + 1) * rank_size] for i in range(num_groups)]

lcoc_time_us = 0
oper_time_us = 0
lcoc_range = []
oper_range = []
# 总共两个group，第一组是baseline_lcoc, 第二组是operator
for i, group in enumerate(groups):
    results = []
    
    # 一个file_path对应一个rank的op_summary文件
    all_last_five_rows = []
    for file_path in group:
        if not os.path.exists(file_path):
            print(f"[{file_path}] File not found.")
            continue
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            if i == 0:# lcoc中allgather跑一次有两行
                ai_core_times = []
                # 检查Task Type是否为AI_CORE，并且Task Duration(us)存在
                task_type_to_check = 'AI_CORE'
                if matmul_type == SPECIAL:
                    task_type_to_check = 'MIX_AIC'
                for row in reader:
                    if (row.get('Task Type') == task_type_to_check and 
                        'Task Duration(us)' in row and 
                        row['Task Duration(us)'].strip()):  
                        ai_core_times.append(float(row['Task Duration(us)']))
                last_five_rows = ai_core_times[-5:]
                all_last_five_rows.append(last_five_rows)
            else: 
                time_list = [float(row['Task Duration(us)']) for row in reader if 'Task Duration(us)' in row]
                last_five_rows = time_list[-5:]
                all_last_five_rows.append(last_five_rows)

    # 取每个rank的最小值，作为算子的性能
    time_us = [min(elements) for elements in zip(*all_last_five_rows)]
    range_value = [round((max(elements) - min(elements)), 4) for elements in zip(*all_last_five_rows)]

    # 五组算子性能的平均值作为算子性能
    if i == 0:
        lcoc_time_us = sum(time_us) / len(time_us)
        lcoc_range.append(range_value)
    elif i == 1:
        oper_time_us = sum(time_us) / len(time_us)
        oper_range.append(range_value)

# 打印 baseline_lcoc 和 operator 的最大时间
if matmul_type == SPECIAL:
    print(f"M:{M} N:{N} K:{K} group_size:{group_size} rankSize:{rank_size} transB:{transB} oper_time:{oper_time_us:.4f} oper_range:{oper_range[0]} "
          f"lcoc_time:{lcoc_time_us:.4f} lcoc_range:{lcoc_range[0]}")
else:
    print(f"M:{M} N:{N} K:{K} rankSize:{rank_size} transB:{transB} oper_time:{oper_time_us:.4f} oper_range:{oper_range[0]} "
          f"lcoc_time:{lcoc_time_us:.4f} lcoc_range:{lcoc_range[0]}")
