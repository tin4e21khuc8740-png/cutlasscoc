1. 使能环境变量

```bash
source /your/cann/path/ascend-toolkit/set_env.sh
source /your/shmem/path/install/set_env.sh
# 运行基线所需的环境变量
source /your/cann/path/nnal/atb/set_env.sh --cxx_abi=0
export PATH=/your/mpich/path/bin:$PATH
export LD_LIBRARY_PATH=/your/mpich/path/lib:$LD_LIBRARY_PATH
export MPI_HOME=/your/mpich/path/
```

2. 编译算子

```bash
cd [代码仓路径]
bash scripts/build.sh 40_allgather_matmul
```

3. 执行算子并比较结果

```bash
cd [代码仓路径]
# OPID为算子编号
# transB 要与.cpp文件中矩阵B的行列优先对应
./run.sh $RANKSIZE $OPID $M $N $K $transB error

# 样例
./run.sh 16 40 33 5555 7777 0 error
```

| 算子编号 | kernel |
| 40 | 40_allgather_matmul |
| 41 | 41_matmul_allreduce |
| 42 | 42_matmul_reducescatter |
| 43 | 43_grouped_matmul_alltoallv |

运行结果：
OP:40_allgather_matmul, M:33, N:555, K:777, rankSize:16, transA:0, transB:0, errorCounts:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

4. 测试性能

在[代码仓路径]/3rdparty下拷贝基线仓

```bash
cd [代码仓路径]/3rdparty
git clone https://gitee.com/zhangyunsong3/synopic-baseline.git
```

编译基线 aclnnMatmul+lccl和lcoc

```bash
cd [代码仓路径]
bash make_baseline.sh
```

# 运行测试性能脚本

每个算子默认执行10次，取后5次平均值作为测试结果

```bash
cd [代码仓路径]
# transB 要与.cpp文件中矩阵B的行列优先对应
./run.sh $RANKSIZE $OPID $M $N $K $transB prof
# 样例
./run.sh 16 40 33 555 777 0 prof
```

运行结果：
Type:allgather_matmul M:33 N:555 K:777 rank_size:16 lccl_time:59.9132 matmul_time:22.48045 lccl_total_time_us:82.39365000000001 lccl_range:[4.159, 5.34, 5.14, 6.36, 10.922] 
M:33 N:555 K:777 rankSize:16 transB:0 oper_time:48.3130 oper_range:[5.26, 4.06, 4.88, 7.82, 7.38] lcoc_time:77.0774 lcoc_range:[4.221, 4909.659, 2.44, 3.019, 3.62]

5. 批量精度测试

进行批量测试前需要编译并重命名可执行文件。

.cpp文件中指定LayoutA/B为RowMajor，编译算子并重命名为{OPNAME}_0_0。指定LayoutB为ColumnMajor，编译并重命名为{OPNAME}_0_1。如下所示：

```bash
output/bin
├── 40_allgather_matmul_0_0
└── 40_allgather_matmul_0_1
```

```bash
cd [代码仓路径]
python3 scripts/batch_error.py $times $OPID 
# 运行样例
python3 ./scripts/batch_error.py 10 40
```

批量测试结果保存在 [代码仓路径]/result/error_data/batch_error.csv

6. 批量性能测试

进行批量测试前需要编译并重命名可执行文件。

.cpp文件中指定LayoutA/B为RowMajor，编译算子并重命名为{OPNAME}_0_0。指定LayoutB为ColumnMajor，编译并重命名为{OPNAME}_0_1。如下所示：

```bash
output/bin
├── 40_allgather_matmul_0_0
└── 40_allgather_matmul_0_1
```

```bash
cd [代码仓路径]

python3 scripts/batch_prof.py $times $OPID
# 运行样例
python3 ./scripts/batch_prof.py 10 40
```

批量测试结果保存在 [代码仓路径]/result/prof_data/batch_prof.csv