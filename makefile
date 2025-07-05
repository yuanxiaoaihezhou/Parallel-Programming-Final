# 1. Compilers and Flags
# ==================================================
CXX = mpic++
NVCC = nvcc

# C++ flags for compiling C++ files (e.g., main.cpp)
# -O3 for optimization, -fopenmp for OpenMP, -mavx2/-mfma for SIMD
CXXFLAGS = -fopenmp -mavx2 -mfma -std=c++17

# NVCC flags for compiling CUDA files (e.g., cuda.cu)
# -O3 for optimization
# -arch=sm_86 is for your RTX 3090 (Ampere architecture)
# -Xcompiler='...' passes flags to the host compiler (g++) that nvcc uses internally
NVCCFLAGS = -arch=sm_86 -Xcompiler="$(CXXFLAGS)"

# Linker flags
LDFLAGS = -fopenmp
# Libraries to link against (CUDA runtime)
LDLIBS = -lcudart


# 2. Source Files and Targets
# ==================================================
TARGET = gaussian_elimination_test
# List of C++ object files
CXX_OBJS = main.o
# List of CUDA object files
CUDA_OBJS = cuda.o
# All object files
OBJS = $(CXX_OBJS) $(CUDA_OBJS)


# 3. Build Rules
# ==================================================
# Default target: build the final executable
all: $(TARGET)

# Rule to link all object files into the final executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS) $(LDLIBS)

# Rule to compile C++ source files (.cpp) into object files (.o)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile CUDA source files (.cu) into object files (.o)
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Rule to clean up build files
clean:
	rm -f *.o $(TARGET) performance_results.csv

.PHONY: all clean