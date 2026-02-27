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
bash scripts/build.sh 41_matmul_allreduce
```

3. 执行算子并比较结果

```bash
cd [代码仓路径]
# OPID为算子编号
# transB 要与.cpp文件中矩阵B的行列优先对应
./run.sh $RANKSIZE $OPID $M $N $K $transB error

# 样例
./run.sh 16 41 33 555 777 0 error
```

| 算子编号 | kernel |
| 40 | 40_allgather_matmul |
| 41 | 41_matmul_allreduce |
| 42 | 42_matmul_reducescatter |
| 43 | 43_grouped_matmul_alltoallv |

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
./run.sh 16 41 33 555 777 0 prof
```

<!-- 运行结果：
Type:matmul_allreduce M:33 N:555 K:777 rank_size:8 lccl_time:8.7882 matmul_time:217.804 lccl_total_time_us:226.5922 lccl_range:[0.539, 1.06, 1.361, 0.12, 0.34] 
M:33 N:555 K:777 rankSize:2 transA:0 transB:0 oper_time:21.9924 oper_range:[0.62, 0.98, 1.239, 0.24, 1.081] lcoc_time:52.6290 lcoc_range:[2.52, 0.02, 0.02, 0.04, 0.56] -->

5. 批量精度测试

进行批量测试前需要编译并重命名可执行文件。

.cpp文件中指定LayoutA/B为RowMajor，编译算子并重命名为{OPNAME}_0_0。指定LayoutB为ColumnMajor，编译并重命名为{OPNAME}_0_1。如下所示：

```bash
output/bin
├── 41_matmul_allreduce_0_0
└── 41_matmul_allreduce_0_1
```

```bash
cd [代码仓路径]
python3 scripts/batch_error.py $times $OPID 
# 运行样例
python3 ./scripts/batch_error.py 10 41
```

批量测试结果保存在 [代码仓路径]/result/error_data/batch_error.csv

6. 批量性能测试

进行批量测试前需要编译并重命名可执行文件。

.cpp文件中指定LayoutA/B为RowMajor，编译算子并重命名为{OPNAME}_0_0。指定LayoutB为ColumnMajor，编译并重命名为{OPNAME}_0_1。如下所示：

```bash
output/bin
├── 41_matmul_allreduce_0_0
└── 41_matmul_allreduce_0_1
```

```bash
cd [代码仓路径]

python3 scripts/batch_prof.py $times $OPID
# 运行样例
python3 ./scripts/batch_prof.py 10 41
```

批量测试结果保存在 [代码仓路径]/result/prof_data/batch_prof.csv