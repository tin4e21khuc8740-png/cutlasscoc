#做批量测试时  生成指定次数的随机M N K   保存在params文件夹下  格式是csv
import sys
import numpy as np
import csv
import os

def gen_test_data(times, OP_ID):
    if times < 0:
        print("times must be greater than or equal to 0!")
        sys.exit(1)

    # 保存M N K的路径
    test_dim_csv_filepath = f"./params/MNK_data.csv"

    # 检查并创建 result 目录
    os.makedirs(os.path.dirname(test_dim_csv_filepath), exist_ok=True)

    # 随机数种子
    np.random.seed(0)
    # 生成随机矩阵
    if OP_ID == 43:
        low_M = 1
        high_M = 128
        low_N = 1024
        high_N = 10240
        low_K = 1024
        high_K = 10240
        low_groupSize = 4
        high_groupSize = 16

        with open(test_dim_csv_filepath, "w") as f_output:
            f_output.write("M,N,K,group_size,rank_size,transA,transB\n")

        # 生成数据并写入 CSV 文件
        for i in range(times):
            M = np.random.randint(low_M, high_M, dtype=np.int32)
            N = np.random.randint(low_N, high_N, dtype=np.int32)
            K = np.random.randint(low_K, high_K, dtype=np.int32)
            group_size = np.random.randint(low_groupSize, high_groupSize, dtype=np.int32)
            # transA = np.random.randint(0, 2)
            # transB = np.random.randint(0, 2)
            transA = 0
            # transB = 0
            transB = np.random.randint(0, 2)
            r = 4
            ranksize = 2 ** r

            with open(test_dim_csv_filepath, "a") as f_output:
                writer = csv.writer(f_output)
                writer.writerow([M, N, K, group_size, ranksize, transA, transB])
            
    else:
        low_M = 1
        high_M = 128
        low_N = 1024
        high_N = 10240
        low_K = 1024
        high_K = 10240

        with open(test_dim_csv_filepath, "w") as f_output:
            f_output.write("M,N,K,rank_size,transA,transB\n")

        # 生成数据并写入 CSV 文件
        for i in range(times):
            M = np.random.randint(low_M, high_M, dtype=np.int32)
            N = np.random.randint(low_N, high_N, dtype=np.int32)
            K = np.random.randint(low_K, high_K, dtype=np.int32)
            transA = 0
            # transB = 0
            transB = np.random.randint(0, 2)
            # r = 3
            r = 4
            ranksize = 2 ** r

            with open(test_dim_csv_filepath, "a") as f_output:
                writer = csv.writer(f_output)
                writer.writerow([M, N, K, ranksize, transA, transB])


if __name__ == "__main__":
    times = int(sys.argv[1])
    OP_ID = int(sys.argv[2])

    gen_test_data(times, OP_ID)