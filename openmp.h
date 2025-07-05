#ifndef OPENMP_H
#define OPENMP_H

#include "baseline.h"
#include <immintrin.h>
#include <omp.h>

// SIMD + OpenMP (static schedule)
void gaussian_elimination_omp_static(matrix& m) {
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

        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; ++k) {
            if (i == k) continue;
            float factor = m[k][i];
            __m256 factor_vec = _mm256_set1_ps(factor);
            
            int j = i;
            for (; j + 8 <= n + 1; j += 8) {
                __m256 row_i_vec = _mm256_loadu_ps(&m[i][j]);
                __m256 row_k_vec = _mm256_loadu_ps(&m[k][j]);
                __m256 res_vec = _mm256_fmsub_ps(factor_vec, row_i_vec, row_k_vec);
                _mm256_storeu_ps(&m[k][j], _mm256_sub_ps(_mm256_setzero_ps(), res_vec));
            }
            for (; j < n + 1; ++j) {
                m[k][j] -= factor * m[i][j];
            }
        }
    }
}

// SIMD + OpenMP (dynamic schedule)
void gaussian_elimination_omp_dynamic(matrix& m) {
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

        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < n; ++k) {
            if (i == k) continue;
            float factor = m[k][i];
            __m256 factor_vec = _mm256_set1_ps(factor);
            
            int j = i;
            for (; j + 8 <= n + 1; j += 8) {
                __m256 row_i_vec = _mm256_loadu_ps(&m[i][j]);
                __m256 row_k_vec = _mm256_loadu_ps(&m[k][j]);
                __m256 res_vec = _mm256_fmsub_ps(factor_vec, row_i_vec, row_k_vec);
                _mm256_storeu_ps(&m[k][j], _mm256_sub_ps(_mm256_setzero_ps(), res_vec));
            }
            for (; j < n + 1; ++j) {
                m[k][j] -= factor * m[i][j];
            }
        }
    }
}

#endif // OPENMP_H