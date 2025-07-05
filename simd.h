#ifndef SIMD_H
#define SIMD_H

#include "baseline.h"
#include <immintrin.h> // For AVX

// SIMD (AVX) 优化的高斯消去
inline void gaussian_elimination_simd(matrix& m) {
    int n = m.size();
    for (int i = 0; i < n; ++i) {
        int max_row = i;
        for (int k = i + 1; k < n; ++k) {
            if (abs(m[k][i]) > abs(m[max_row][i])) {
                max_row = k;
            }
        }
        std::swap(m[i], m[max_row]);

        float pivot = m[i][i];
        for (int j = i; j < n + 1; ++j) {
            m[i][j] /= pivot;
        }
        
        for (int k = 0; k < n; ++k) {
            if (i == k) continue;
            float factor = m[k][i];
            __m256 factor_vec = _mm256_set1_ps(factor);
            
            int j = i;
            // AVX 一次处理8个float
            for (; j + 8 <= n + 1; j += 8) {
                __m256 row_i_vec = _mm256_loadu_ps(&m[i][j]);
                __m256 row_k_vec = _mm256_loadu_ps(&m[k][j]);
                __m256 res_vec = _mm256_fmsub_ps(factor_vec, row_i_vec, row_k_vec); // res = -(factor * row_i - row_k)
                _mm256_storeu_ps(&m[k][j], _mm256_sub_ps(_mm256_setzero_ps(), res_vec)); // Negate the result back
            }
            
            // 处理剩余的不足8个的元素
            for (; j < n + 1; ++j) {
                m[k][j] -= factor * m[i][j];
            }
        }
    }
}

#endif // SIMD_H