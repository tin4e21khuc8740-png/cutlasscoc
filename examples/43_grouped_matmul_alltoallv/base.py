import numpy as np
import os
import sys

def read_bin_file(filename, dtype=np.float32):
    """读取二进制文件并返回numpy数组"""
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=dtype)
    return data

def matrix_grouped_matmul_alltoallv(M, N, K, ranksize, transB, startid, group_size, dtype=np.float16):
    """
    读取多组a_x.bin和b_x.bin，进行分组矩阵乘法并alltoallv结果
    """

    global_tokens_per_expert_file = f"./examples/27_grouped_matmul_alltoallv/data/globalTokensPerExpert.bin"
    if not os.path.exists(global_tokens_per_expert_file):
        print(f"文件 {global_tokens_per_expert_file} 不存在，无法进行 grouped matmul alltoallv")
        sys.exit(1)
    globalTokensPerExpert = read_bin_file(global_tokens_per_expert_file, dtype=np.uint32).reshape((ranksize, group_size * ranksize))

    # 先计算所有rank的WC，并紧密地拼接到一起，最后再根据表生成所有rank的最终结果
    expected_wc_all_ranks = []
    for i in range(ranksize):
        a_file = f"./examples/27_grouped_matmul_alltoallv/data/a{startid + i}.bin"
        b_file = f"./examples/27_grouped_matmul_alltoallv/data/b{startid + i}.bin"

        if not os.path.exists(a_file) or not os.path.exists(b_file):
            print(f"文件 {a_file} 或 {b_file} 不存在，无法进行 grouped matmul alltoallv")
            sys.exit(1)

        offset_in_a = 0
        offset_in_b = 0
        row_offset_in_wc = 0

        for group in range(group_size):
            global_expert_idx = i * group_size + group

            m_in_problem = globalTokensPerExpert[:, global_expert_idx].sum()

            elements_to_read_a = m_in_problem * K
            elements_to_read_b = K * N

            flat_a = np.fromfile(a_file, dtype=np.float16, count=elements_to_read_a, offset=offset_in_a * np.dtype(np.float16).itemsize)
            flat_b = np.fromfile(b_file, dtype=np.float16, count=elements_to_read_b, offset=offset_in_b * np.dtype(np.float16).itemsize)
            
            sub_a = flat_a.reshape((m_in_problem, K))

            if(transB):
                sub_b = flat_b.reshape((K, N), order='F')
            else:
                sub_b = flat_b.reshape((K, N))
            
            # 计算期望结果 (fp16*fp16 → fp32 accumulate → cast fp16)
            sub_expected_wc = np.matmul(sub_a.astype(np.float32), sub_b.astype(np.float32)).astype(np.float16)

            # 把sub_expected_wc拼接到所有rank的expected_wc_all_ranks里
            expected_wc_all_ranks.append(sub_expected_wc)  

            # 偏移更新
            offset_in_a += elements_to_read_a
            offset_in_b += elements_to_read_b
            row_offset_in_wc += m_in_problem
    
    expected_wc_all_ranks = np.vstack(expected_wc_all_ranks)


    # 根据表，拼凑出所有rank的expected_c_all_ranks
    expected_c_all_ranks = np.zeros((ranksize * M, N), dtype=np.float16)
    # 一个用于记录各个 rank 上 C 的行偏移的数组
    row_offset_in_c = np.zeros(ranksize, dtype=np.int32)
    # WC 上的行偏移
    row_offset_in_wc_all_ranks = 0

    # 按列遍历整张表
    for expert_idx in range(ranksize * group_size):
        for rank in range(ranksize):
            token_num = globalTokensPerExpert[rank, expert_idx]
            if token_num == 0:
                continue

            # 从 expectedwc 里取出对应的 token 行
            sub_wc = expected_wc_all_ranks[row_offset_in_wc_all_ranks : row_offset_in_wc_all_ranks + token_num, :]

            # 写到对应 rank 的 C
            start = rank * M + row_offset_in_c[rank]
            end = start + token_num
            expected_c_all_ranks[start:end, :] = sub_wc

            # 更新偏移
            row_offset_in_c[rank] += token_num
            row_offset_in_wc_all_ranks += token_num

# 遍历所有rank的C，分别写回文件
    for rank in range(ranksize):
        result_file = f"./examples/27_grouped_matmul_alltoallv/data/golden_result{startid + rank}.bin"
        
        rows = M
        subResult = expected_c_all_ranks[rank * rows : (rank + 1)* rows, :]
        subResult.tofile(result_file)

if __name__ == "__main__":
    
    M = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    ranksize = int(sys.argv[4])
    transB = int(sys.argv[5])
    startid = int(sys.argv[6])
    group_size = int(sys.argv[7])
    matrix_grouped_matmul_alltoallv(M, N, K, ranksize, transB, startid, group_size)