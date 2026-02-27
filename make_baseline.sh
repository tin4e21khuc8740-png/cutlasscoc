# 编译aclnnMm
cd ./3rdparty/aclnnMatmul
bash make.sh

cd -

# 编译aclnnGroupedMatmul
cd ./3rdparty/aclnnGroupedMatmul
bash make.sh

cd -

# 编译lccl和lcoc
cd ./3rdparty/synopic-baseline
mkdir build
cmake -B build -S .
cmake --build build --target all

# 编译hccl  
cd $ASCEND_HOME_PATH/tools/hccl_test
make MPI_HOME=/usr/local/mpich ASCEND_DIR=$ASCEND_HOME_PATH -j