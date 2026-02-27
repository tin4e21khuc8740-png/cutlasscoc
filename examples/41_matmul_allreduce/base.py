import numpy as np
import os
import sys

def read_bin_file(filename, dtype=np.float32):
    """读取二进制文件并返回numpy数组"""
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=dtype)
    return data

def matrix_multiply_and_reduce(M, N, K, ranksize, trans, startid, dtype=np.float16):
    """
    读取多组a_x.bin和b_x.bin，进行矩阵乘法并reduce结果
    
    参数:
        num_files: 文件组数 (x从0到num_files-1)
        shape_a: 矩阵a的形状 (rows_a, cols_a)
        shape_b: 矩阵b的形状 (rows_b, cols_b)
        dtype: 数据类型，默认为float32
    """
    # 初始化结果矩阵
    result = np.zeros((M, N), dtype=np.float16)
    
    for i in range(ranksize):
        # 构造文件名
        a_file = f"./examples/41_matmul_allreduce/data/a{startid + i}.bin"
        b_file = f"./examples/41_matmul_allreduce/data/b{startid + i}.bin"
        
        # 检查文件是否存在
        if not os.path.exists(a_file) or not os.path.exists(b_file):
            print(f"文件 {a_file} 或 {b_file} 不存在，跳过")
            continue
        
        # 读取数据
        # if trans_a:
        #     a = read_bin_file(a_file, dtype).reshape((M,K), order="F")
        # else:
        a = read_bin_file(a_file, dtype).reshape((M,K))

        if trans:  
            b = read_bin_file(b_file, dtype).reshape((K,N), order="F")
        else:
            b = read_bin_file(b_file, dtype).reshape((K,N))
        
        # 矩阵乘法
        partial_result = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)

        # partial_result.tofile(f"./examples/41_matmul_allreduce/data/golden_result{i}.bin")
        
        
        # 累加到总结果
        result += partial_result
        # result = partial_result

        # print("rank", i, result[0])
        
        print(f"已处理 {a_file} 和 {b_file}")
    
    return result

# 示例用法
if __name__ == "__main__":
    
    M = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    ranksize = int(sys.argv[4])
    # trans_a = int(sys.argv[5])
    trans_b = int(sys.argv[5])
    startid = int(sys.argv[6])
    
    final_result = matrix_multiply_and_reduce(M, N, K, ranksize, trans_b, startid)
    
    # 保存最终结果
    final_result.tofile("./examples/41_matmul_allreduce/data/golden_result.bin")
    