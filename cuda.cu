#include <iostream>
#include <vector>
#include "cuda.h"

// 定义矩阵类型
using matrix = std::vector<std::vector<float>>;

// CUDA 核函数，用于并行处理每一行的消元
__global__ void elimination_kernel(float* m_dev, int n, int i, float* pivot_row) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < n && k != i) {
        float factor = m_dev[k * (n + 1) + i];
        for (int j = i; j < n + 1; ++j) {
            m_dev[k * (n + 1) + j] -= factor * pivot_row[j];
        }
    }
}

// CUDA 主函数
void gaussian_elimination_cuda(matrix& m) {
    int n = m.size();
    int matrix_elements = n * (n + 1);
    size_t matrix_size_bytes = matrix_elements * sizeof(float);
    size_t row_size_bytes = (n + 1) * sizeof(float);

    // 1. 在GPU上分配内存
    float* m_dev;
    cudaMalloc(&m_dev, matrix_size_bytes);

    // 将二维vector转为一维数组以便传输
    std::vector<float> m_flat;
    m_flat.reserve(matrix_elements);
    for(const auto& row : m) {
        m_flat.insert(m_flat.end(), row.begin(), row.end());
    }

    // 2. 将数据从CPU拷贝到GPU
    cudaMemcpy(m_dev, m_flat.data(), matrix_size_bytes, cudaMemcpyHostToDevice);

    // 为主元行在GPU上分配内存
    float* pivot_row_dev;
    cudaMalloc(&pivot_row_dev, row_size_bytes);
    
    // 3. 高斯消去主循环 (在CPU中控制)
    for (int i = 0; i < n; ++i) {
        // (寻找主元在GPU上做很低效，通常在CPU上完成)
        // 这里为了简化，我们假设主元就是m[i][i]且不为0
        // 一个完整的实现需要将列数据拷回CPU找主元，再交换行指针
        
        // 将主元行拷贝到GPU上的专用内存
        cudaMemcpy(pivot_row_dev, m_dev + i * (n + 1), row_size_bytes, cudaMemcpyDeviceToDevice);
        
        // 归一化主元行 (可以在一个简单的kernel里做，或者就在CPU上算完再传)
        // 这里为了简化，我们依然在CPU上操作，然后把更新后的主元行传回去
        // 这不是最高效的，但能展示基本思路
        std::vector<float> temp_row(n + 1);
        cudaMemcpy(temp_row.data(), pivot_row_dev, row_size_bytes, cudaMemcpyDeviceToHost);
        float pivot_val = temp_row[i];
        for(int j=i; j<n+1; ++j) {
            temp_row[j] /= pivot_val;
        }
        cudaMemcpy(pivot_row_dev, temp_row.data(), row_size_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(m_dev + i * (n+1), temp_row.data(), row_size_bytes, cudaMemcpyHostToDevice);
        

        // 设置kernel启动参数
        int threads_per_block = 256;
        int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

        // 4. 调用kernel执行并行消元
        elimination_kernel<<<blocks_per_grid, threads_per_block>>>(m_dev, n, i, pivot_row_dev);

        // 同步以确保kernel执行完毕
        cudaDeviceSynchronize();
    }
    
    // 5. 将结果从GPU拷回CPU
    cudaMemcpy(m_flat.data(), m_dev, matrix_size_bytes, cudaMemcpyDeviceToHost);

    // 6. 释放GPU内存
    cudaFree(m_dev);
    cudaFree(pivot_row_dev);

    // 将一维数组结果转回二维vector
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n + 1; ++j) {
            m[i][j] = m_flat[i * (n + 1) + j];
        }
    }
}

// C++ Wrapper, 这是 main.cpp 会调用的函数
void gaussian_elimination_cuda_wrapper(matrix& m) {
    gaussian_elimination_cuda(m);
}