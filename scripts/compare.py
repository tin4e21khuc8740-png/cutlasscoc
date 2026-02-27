import numpy as np
import os
import sys

def read_file(file_path):
    data = np.fromfile(file_path, dtype=np.float16)
    return data

def find_top10(arr):
    flat_idx = np.argsort(arr.ravel())[-10:][::-1]
    coords = np.unravel_index(flat_idx, arr.shape)
    values = arr.ravel()[flat_idx]
    return values, list(zip(*coords))

def compute_error(expected_result, result):
    expected_result = np.array(expected_result).flatten()
    result = np.array(result).flatten()
    if len(expected_result) != len(result):
        raise ValueError("两个数组的长度不同")
    
    # 绝对误差
    abs_error = np.abs(result - expected_result)
    # 相对误差
    rel_error = abs_error / (np.abs(expected_result) + 1e-7)
    # 最大相对误差
    MARE = np.max(rel_error)
    # 平均相对误差
    MERE = np.mean(rel_error)
    # 均方根误差
    RMSE = np.sqrt(np.mean((result - expected_result) ** 2))
    # 误差均衡性
    EB = np.mean((result - expected_result) / np.maximum(np.abs(expected_result), 1.0))
    return abs_error, rel_error, MARE, MERE, RMSE, EB

def run(operator_name, m, n, k, rank_size, transA, transB, start_id):
    directory_path = f"./examples/{operator_name}/data"

    # 计算golden数据并与result对比
    print("Verifying the final results...")
    summary_error_count = []
    expected_data = []
    if transA == 0:
        # python实现allgather
        for i in range(rank_size):
            file_a = read_file(f"{directory_path}/a{start_id + i}.bin")
            expected_data.append(file_a)
        expected_data_1D = np.concatenate(expected_data)

        a_matrix = expected_data_1D.reshape(rank_size * m, k)
        # print(f"!!!!!!!a_matrix:\n{a_matrix}")
    else:
        for i in range(rank_size):
            file_a = read_file(f"{directory_path}/a{start_id + i}.bin")
            a_local_matrix = file_a.reshape(m, k, order = 'F')
            expected_data.append(a_local_matrix)
        # print(expected_data)
        a_matrix = np.concatenate(expected_data, axis=0)
        # print(f"!!!!!!!a_matrix:\n{a_matrix}")

    for i in range(rank_size):
        print("-" * 80)
        result = read_file(f"{directory_path}/result{start_id + i}.bin")

        b_matrix_1D = read_file(f"{directory_path}/b{start_id + i}.bin")
        if transB == 0:
            b_matrix = b_matrix_1D.reshape(k, n)
            # print(f"!!!!!!!b_matrix:\n{b_matrix}")
        else:
            b_matrix = b_matrix_1D.reshape(k, n, order='F')
            # b_matrix = b_matrix_1D.reshape(n, k).T
            # print(f"!!!!!!!b_matrix:\n{b_matrix}")
        c_matrix_fp32 = a_matrix.astype(np.float32) @ b_matrix.astype(np.float32)
        # print(f"!!!!!!!c_matrix_fp32:\n{c_matrix_fp32}")
        golden = np.concatenate(c_matrix_fp32)
        
        # cpu_low = (a_matrix.astype(np.float16) @ b_matrix.astype(np.float16)).flatten()
        
        abs_error_npu, rel_error_npu, MARE_npu, MERE_npu, RMSE_npu, EB_npu = compute_error(golden.astype(np.float16), result)
        # abs_error_cpu, rel_error_cpu, MARE_cpu_low, MERE_cpu_low, RMSE_cpu_low, EB_cpu_low = compute_error(golden, cpu_low)

        # 验收标准
        # if ((MARE_npu / max(MARE_cpu_low, 2e-11)) < 10 and 
        #     (MERE_npu / max(MERE_cpu_low, 2e-11)) < 2 and
        #     (RMSE_npu / max(RMSE_cpu_low, 2e-11)) < 2 and
        #     (abs(EB_npu) < pow(2, -10))):
        #     print(f"[PASS] rank {i} result passes the acceptance standards.")
        # else:
        #     print(f"[FAIL] rank {i} result doesn't pass the acceptance standards.")
        #     print(f"  NPU: MARE={MARE_npu}, MERE={MERE_npu}, RMSE={RMSE_npu}, EB={EB_npu}")
        #     print(f"  CPU: MARE={MARE_cpu_low}, MERE={MERE_cpu_low}, RMSE={RMSE_cpu_low}, EB={EB_cpu_low}")
        #     print(f"  MARE compare: {(MARE_npu / max(MARE_cpu_low, 2e-11))}, MERE compare: {(MERE_npu / max(MERE_cpu_low, 2e-11))}, RMSE compare: {(RMSE_npu / max(RMSE_cpu_low, 2e-11))}")

        # 输出相对、绝对误差都大于千分之一的元素及坐标
        is_equal = (rel_error_npu <= 0.001) | (abs_error_npu <= 0.001)
        if np.all(is_equal):
            print(f"[COMPARE SUCCESS] rank {start_id + i} every element is equal to golden.")
            print("errorCount: 0")
            summary_error_count.append(0)
        else:
            diff_indices = np.where(~is_equal)[0]
            num_diffs = min(10, len(diff_indices))
            print(f"[COMPARE FAIL] rank {start_id + i} every element is not equal to golden. Differences at indices:")
            for j in range(num_diffs):
                idx = diff_indices[j]
                print(f"  Index {idx}: expected {golden[idx].astype(np.float16)}, actual {result[idx]}")
            print(f"errorCount: {len(diff_indices)}")
            summary_error_count.append(len(diff_indices))
        # print(f"!!!!!!!a_matrix:\n{a_matrix}")
        # print(f"!!!!!!!b_matrix:\n{b_matrix}")
        # print(f"expected[0]: {golden[0]}, result[0]: {result[0]}")
    
        # # 输出top10相对误差及坐标
        # top_vals_npu, top_coords_npu = find_top10(rel_error_npu)
        # top_vals_cpu, top_coords_cpu = find_top10(rel_error_cpu)
        # print(f"Top 10 relative errors for rank {i} (NPU):")
        # for i, (val, coord) in enumerate(zip(top_vals_npu, top_coords_npu), 1):
        #     print(f"  Index {coord[0]}: expected {golden[coord]}, actual {result[coord]}, rel_error {val}")
        # print(f"Top 10 relative errors for CPU:")
        # for i, (val, coord) in enumerate(zip(top_vals_cpu, top_coords_cpu), 1):
        #     print(f"  Index {coord[0]}: expected {golden[coord]}, actual {cpu_low[coord]}, rel_error {val}")
    print(f"OP:{operator_name}, M:{m}, N:{n}, K:{k}, rankSize:{rank_size}, transA:{transA}, transB:{transB}, errorCounts:{summary_error_count}")

def compare_all_reduce(m, n, k, rank_size, transA, transB, start_id):
    directory_path = f"./examples/{operator_name}/data"

    golden = read_file(f"{directory_path}/golden_result.bin")
    print("golden:", golden[0])

    summary_error_count = []
    for i in range(rank_size):
        print("-" * 80)
        # golden = read_file(f"{directory_path}/golden_result{i}.bin")
        result = read_file(f"{directory_path}/result{start_id + i}.bin")
        
        # cpu_low = (a_matrix.astype(np.float16) @ b_matrix.astype(np.float16)).flatten()
        
        # 计算四个指标参数
        abs_error_npu, rel_error_npu, MARE_npu, MERE_npu, RMSE_npu, EB_npu = compute_error(golden, result)
        # abs_error_cpu, rel_error_cpu, MARE_cpu_low, MERE_cpu_low, RMSE_cpu_low, EB_cpu_low = compute_error(golden, cpu_low)

        # 输出相对、绝对误差都大于千分之一的元素及坐标
        is_equal = (rel_error_npu <= 0.005) | (abs_error_npu <= 0.005)
        if np.all(is_equal):
            print(f"[COMPARE SUCCESS] rank {start_id + i} every element is equal to golden.")
            print("errorCount: 0")
            summary_error_count.append(0)
            # num_diffs = 10
            # for j in range(num_diffs):
            #     print(f"  Index {j}: expected {golden[j].astype(np.float16)}, actual {result[j]}")
        else:
            diff_indices = np.where(~is_equal)[0]
            num_diffs = min(10, len(diff_indices))
            print(f"[COMPARE FAIL] rank {start_id + i} every element is not equal to golden. Differences at indices:")
            for j in range(num_diffs):
                idx = diff_indices[j]
                print(f"  Index {idx}: expected {golden[idx].astype(np.float16)}, actual {result[idx]}")
            print(f"errorCount: {len(diff_indices)}")
            summary_error_count.append(len(diff_indices))
    print(f"OP:{operator_name}, M:{m}, N:{n}, K:{k}, rankSize:{rank_size}, transA:{transA}, transB:{transB}, errorCounts:{summary_error_count}")

def compare_reduce_scatter(operator_name, m, n, k, rank_size,  transA, transB, start_rank_id):
    directory_path = f"./examples/{operator_name}/data"

    summary_error_count = []
    for i in range(rank_size):
        print("-" * 80)
        # golden = read_file(f"{directory_path}/golden_result{i}.bin")
        try:
            golden = read_file(f"{directory_path}/golden_result{start_rank_id + i}.bin")
        except FileNotFoundError:
            print(f"[ERROR] Golden result file not found: {directory_path}/golden_result{start_rank_id + i}.bin")
            continue # 跳过当前 rank 的比较
        result = read_file(f"{directory_path}/result{start_rank_id + i}.bin")

        if golden.shape != result.shape:
             print(f"[ERROR] Rank {i}: Golden result shape {golden.shape} does not match actual result shape {result.shape}")
             continue
        
        # cpu_low = (a_matrix.astype(np.float16) @ b_matrix.astype(np.float16)).flatten()
        
        # 计算四个指标参数
        abs_error_npu, rel_error_npu, MARE_npu, MERE_npu, RMSE_npu, EB_npu = compute_error(golden, result)
        # abs_error_cpu, rel_error_cpu, MARE_cpu_low, MERE_cpu_low, RMSE_cpu_low, EB_cpu_low = compute_error(golden, cpu_low)

        # 输出相对、绝对误差都大于千分之一的元素及坐标
        is_equal = (rel_error_npu <= 0.005) | (abs_error_npu <= 0.005)
        if np.all(is_equal):
            print(f"[COMPARE SUCCESS] rank {start_rank_id + i} every element is equal to golden.")
            print("errorCount: 0")
            summary_error_count.append(0)
            # num_diffs = 10
            # for j in range(num_diffs):
            #     print(f"  Index {j}: expected {golden[j].astype(np.float16)}, actual {result[j]}")
        else:
            diff_indices = np.where(~is_equal)[0]
            num_diffs = min(10, len(diff_indices))
            print(f"[COMPARE FAIL] rank {start_rank_id + i} every element is not equal to golden. Differences at indices:")
            for j in range(num_diffs):
                idx = diff_indices[j]
                print(f"  Index {idx}: expected {golden[idx].astype(np.float16)}, actual {result[idx]}")
            print(f"errorCount: {len(diff_indices)}")
            summary_error_count.append(len(diff_indices))
    print(f"OP:{operator_name}, M:{m}, N:{n}, K:{k}, rankSize:{rank_size}, transA:{transA}, transB:{transB}, errorCounts:{summary_error_count}")

def compare_alltoallv(operator_name, m, n, k, rank_size, transA, transB, start_id, group_size):
    directory_path = f"./examples/{operator_name}/data"

    summary_error_count = []
    for i in range(rank_size):
        print("--------------------------------------------------------------------------------------------------------------------------")
        try:
            golden = read_file(f"{directory_path}/golden_result{start_id + i}.bin")
        except FileNotFoundError:
            print(f"[ERROR] Golden result file not found: {directory_path}/golden_result{start_id + i}.bin")
            continue # 跳过当前 rank 的比较
        result = read_file(f"{directory_path}/result{start_id + i}.bin")

        if golden.shape != result.shape:
             print(f"[ERROR] Rank {i}: Golden result shape {golden.shape} does not match actual result shape {result.shape}")
             continue
        
        # cpu_low = (a_matrix.astype(np.float16) @ b_matrix.astype(np.float16)).flatten()
        
        # 计算四个指标参数
        abs_error_npu, rel_error_npu, MARE_npu, MERE_npu, RMSE_npu, EB_npu = compute_error(golden, result)
        # abs_error_cpu, rel_error_cpu, MARE_cpu_low, MERE_cpu_low, RMSE_cpu_low, EB_cpu_low = compute_error(golden, cpu_low)

        # 输出相对、绝对误差都大于千分之一的元素及坐标
        is_equal = (rel_error_npu <= 0.001) | (abs_error_npu <= 0.001)
        if np.all(is_equal):
            print(f"[COMPARE SUCCESS] rank {i} every element is equal to golden.")
            print("errorCount: 0")
            summary_error_count.append(0)
            # num_diffs = 10
            # for j in range(num_diffs):
            #     print(f"  Index {j}: expected {golden[j].astype(np.float16)}, actual {result[j]}")
        else:
            diff_indices = np.where(~is_equal)[0]
            num_diffs = min(10, len(diff_indices))
            print(f"[COMPARE FAIL] rank {i} every element is not equal to golden. Differences at indices:")
            for j in range(num_diffs):
                idx = diff_indices[j]
                print(f"  Index {idx}: expected {golden[idx].astype(np.float16)}, actual {result[idx]}")
            print(f"errorCount: {len(diff_indices)}")
            summary_error_count.append(len(diff_indices))
    print(f"OP:{operator_name}, M:{m}, N:{n}, K:{k}, groupSize:{group_size}, rankSize:{rank_size}, transA:{transA}, transB:{transB}, errorCounts:{summary_error_count}")



if __name__ == "__main__":
    operator_name = sys.argv[1]
    m = int(sys.argv[2])
    n = int(sys.argv[3])
    k = int(sys.argv[4])
    rank_size = int(sys.argv[5])
    transB = int(sys.argv[6])
    start_id = int(sys.argv[7])
    transA = 0
    if operator_name == "40_allgather_matmul":
        run(operator_name, m, n, k, rank_size, transA, transB, start_id)
    elif operator_name == "41_matmul_allreduce":
        compare_all_reduce(m, n, k, rank_size, transA, transB, start_id)
    elif operator_name == "42_matmul_reducescatter":
        compare_reduce_scatter(operator_name, m, n, k, rank_size, transA, transB, start_id)
    elif operator_name == "43_grouped_matmul_alltoallv":
        group_size = int(sys.argv[8])
        compare_alltoallv(operator_name, m, n, k, rank_size, transA, transB, start_id, group_size)
    else:
        print(f"[ERROR] 不支持的算子!")