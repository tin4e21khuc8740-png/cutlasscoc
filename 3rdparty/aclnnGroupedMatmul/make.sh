source /usr/local/Ascend/ascend-toolkit/set_env.sh

rm -rf build 

mkdir build

cd build
cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
make
