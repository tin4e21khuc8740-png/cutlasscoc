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
bash scripts/build.sh 27_grouped_matmul_alltoallv
```

3. 执行算子并比较结果

```bash
cd [代码仓路径]
# OPID为算子编号
# transB 要与.cpp文件中矩阵B的行列优先对应
./run.sh $RANKSIZE $OPID $M $N $K $transB error $group_size

# 样例
./run.sh 8 27 33 555 777 0 error 5
```
| 算子编号 | kernel |
| 24 | 24_allgather_matmul |
| 25 | 25_matmul_allreduce |
| 26 | 26_matmul_reducescatter |
| 27 | 27_grouped_matmul_alltoallv |

运行结果：
OP:27_grouped_matmul_alltoallv, M:33, N:555, K:777, groupSize:5, rankSize:8, transA:0, transB:0, errorCounts:[0, 0, 0, 0, 0, 0, 0, 0]

4. 测试性能

在[代码仓路径]/3rdparty下拷贝基线仓

```bash
cd [代码仓路径]/3rdparty
git clone https://gitee.com/zhangyunsong3/synopic-baseline.git
```

编译基线 aclnnGroupedMatmul+lccl和lcoc

openmpi版本：
    需要将make_baseline_all2allv.sh中，编译hccl的make命令的MPI_HOME替换为openmpi的路径：
        make MPI_HOME=[openmpi-path] ASCNED_DIR=$ASCEND_HOME_PATH -j
mpich版本（即目前版本）：
    需要将./3rdparty/run_hccl.sh中第22行MPI_MPICH参数改为true：
        MPI_MPICH=true # 可选 'true' 或 'false'

```bash
cd [代码仓路径]
bash make_baseline_all2allv.sh
```
# 运行测试性能脚本

每个算子默认执行10次，取后5次平均值作为测试结果

```bash
cd [代码仓路径]
# transB 要与.cpp文件中矩阵B的行列优先对应
./run.sh $RANKSIZE $OPID $M $N $K $transB prof $group_size
# 样例
./run.sh 8 27 33 555 777 0 prof 5
```

运行结果：
Type:GroupedMatmulalltoallv
M:33 N:555 K:777 group_size:5 rankSize:8 hccl_time:118.05 aclnn_time:28.59 total_time:146.64
M:33 N:555 K:777 group_size:5 rankSize:8 transB:0 oper_time:42.8490 oper_range:[1.0, 2.46, 1.42, 2.64, 2.14] lcoc_time:65.5292 lcoc_range:[1.72, 0.74, 0.86, 1.861, 0.84]

5. 批量精度测试

进行批量测试前需要编译并重命名可执行文件。

.cpp文件中指定LayoutA/B为RowMajor，编译算子并重命名为{OPNAME}_0_0。指定LayoutB为ColumnMajor，编译并重命名为{OPNAME}_0_1。如下所示：

```bash
output/bin
├── 27_grouped_matmul_alltoallv_0_0
└── 27_grouped_matmul_alltoallv_0_1
```

```bash
cd [代码仓路径]
python3 scripts/batch_error.py $times $OPID 
# 运行样例
python3 ./scripts/batch_error.py 10 27
```

批量测试结果保存在 [代码仓路径]/result/error_data/batch_error.csv

6. 批量性能测试

进行批量测试前需要编译并重命名可执行文件。

.cpp文件中指定LayoutA/B为RowMajor，编译算子并重命名为{OPNAME}_0_0。指定LayoutB为ColumnMajor，编译并重命名为{OPNAME}_0_1。如下所示：

```bash
output/bin
├── 27_grouped_matmul_alltoallv_0_0
└── 27_grouped_matmul_alltoallv_0_1
```

```bash
cd [代码仓路径]
python3 scripts/batch_prof.py $times $OPID
# 运行样例
python3 ./scripts/batch_prof.py 10 27
```
批量测试结果保存在 [代码仓路径]/result/prof_data/batch_prof.csv

