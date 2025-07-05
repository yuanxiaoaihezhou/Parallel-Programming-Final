#ifndef BASELINE_H
#define BASELINE_H

#include <vector>
#include <iostream>

// 为了方便，我们定义一个矩阵类型
using matrix = std::vector<std::vector<float>>;

// 初始化矩阵 (随机值)
void init_matrix(matrix& m, int size) {
    m.assign(size, std::vector<float>(size + 1, 0));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size + 1; ++j) {
            m[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
}

// 打印矩阵 (用于调试)
void print_matrix(const matrix& m) {
    for (const auto& row : m) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

// 基准高斯消去算法
void gaussian_elimination_baseline(matrix& m) {
    int n = m.size();
    for (int i = 0; i < n; ++i) {
        // 寻找主元
        int max_row = i;
        for (int k = i + 1; k < n; ++k) {
            if (abs(m[k][i]) > abs(m[max_row][i])) {
                max_row = k;
            }
        }
        std::swap(m[i], m[max_row]);

        // 对i行进行归一化
        float pivot = m[i][i];
        for (int j = i; j < n + 1; ++j) {
            m[i][j] /= pivot;
        }

        // 消去其他行的第i列
        for (int k = 0; k < n; ++k) {
            if (i == k) continue;
            float factor = m[k][i];
            for (int j = i; j < n + 1; ++j) {
                m[k][j] -= factor * m[i][j];
            }
        }
    }
}

#endif // BASELINE_H