# 编译器设置
CXX = mpic++
NVCC = nvcc
# 编译标志
# -O3 开启最高级别优化
# -mavx2 明确告诉编译器使用AVX2指令集
# -fopenmp 开启OpenMP支持
CXXFLAGS = -O3 -mavx2 -fopenmp -std=c++17
# CUDA编译标志
# -O3 优化
# -arch=sm_86 针对RTX 3090 (Ampere架构) 进行编译
# --ptxas-options=-v 显示寄存器使用等详细信息
NVCCFLAGS = -O3 -arch=sm_86 --ptxas-options=-v
# 链接库
LDFLAGS = -fopenmp
LDLIBS = -lcudart

# 目标文件
TARGET = gaussian_elimination_test
OBJS = main.o cuda_obj.o

# 默认目标
all: $(TARGET)

# 链接生成最终可执行文件
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS) $(LDLIBS)

# 编译 main.cpp
main.o: main.cpp baseline.h simd.h openmp.h mpi.h cuda.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

# 编译 cuda.cu 并生成一个可被C++链接的目标文件
cuda_obj.o: cuda.cu cuda.h
	$(NVCC) $(NVCCFLAGS) -dc cuda.cu -o cuda_intermediate.o
	$(NVCC) $(NVCCFLAGS) -dlink cuda_intermediate.o -o cuda_link.o
	$(CXX) $(CXXFLAGS) -c cuda.cu -o cuda_obj_temp.o
	ld -r -o cuda_obj.o cuda_intermediate.o cuda_link.o cuda_obj_temp.o

# 清理
clean:
	rm -f *.o $(TARGET) performance_results.csv

.PHONY: all clean