#!/bin/bash

# 清理旧文件
make clean

# 编译项目
make

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "Compilation failed. Aborting."
    exit 1
fi

# 设置OpenMP线程数
# AMD EPYC 7402 是 24 核 48 线程. 我们可以设置为物理核心数 24.
export OMP_NUM_THREADS=24

# 运行MPI程序
# -np 4 使用4个进程. 你可以根据需要调整.
# --use-hwthread-cpus 允许MPI进程使用逻辑核心(超线程)
# --bind-to core 将进程绑定到核心，以获得更好的性能
echo "Running tests with OMP_NUM_THREADS=$OMP_NUM_THREADS and 4 MPI processes..."

mpirun -np 4 --bind-to core ./gaussian_elimination_test

echo "Script finished."