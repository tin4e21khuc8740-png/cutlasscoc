# W4A8_Matmul Example Readme
## 代码组织
```
├── 32_w4a8_matmul
│   ├── CMakeLists.txt # CMake编译文件
│   ├── gen_data.py
│   ├── w4a8.cpp 
│   └── README.md
```
## 功能介绍
- 提供了W4A8量化模式下的matmul实现
- A矩阵int8类型，B矩阵int4类型，对B矩阵cast成int8后，矩阵乘法结果int32，通过per tensor量化为half类型输出
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)   

- 接下来，先执行`gen_data.py`，生成测试样例，测试用例需要从命令行输入, 执行该命令后会在当前路径下生成data目录，包含算子的输入数据和用于精度验证的golden数据。   
- 然后执行算子，这里要注意的是执行算子的输入shape和上面第一步生成数据的shape一致。

以下是一个完整的shell脚本示例（在样例目录`./examples/32_w4a8_matmul`下执行）
```
m=860
k=5712
n=4535
device=0

function build() {
    rm -rf ../../build
    rm -rf ../../output
    bash ../../scripts/build.sh 32_w4a8_matmul
}

function gen_data() {
    python3 gen_data.py $m $n $k
    echo "Data gen finished"
}

function run_kernel {
    echo 'Case: m=' $m ' k=' $k ' n=' $n
    cd ../../output/bin/
    ./32_w4a8_matmul $m $n $k $device
}

build
gen_data
run_kernel
```

执行结果如下，说明精度比对成功。
```
Compare success.
```