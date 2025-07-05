#ifndef MPI_H
#define MPI_H

#include "baseline.h"
#include <immintrin.h>
#include <omp.h>
#include <mpi.h>

// SIMD + OpenMP + MPI
inline void gaussian_elimination_mpi(matrix& m, int rank, int world_size) {
    int n = m.size();

    for (int i = 0; i < n; ++i) {
        // 主进程 (rank 0) 寻找主元并广播
        if (rank == 0) {
            int max_row = i;
            for (int k = i + 1; k < n; ++k) {
                if (abs(m[k][i]) > abs(m[max_row][i])) {
                    max_row = k;
                }
            }
            std::swap(m[i], m[max_row]);
        }
        
        // 广播主元行
        MPI_Bcast(m[i].data(), n + 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        // 每个进程并行处理自己负责的行
        #pragma omp parallel for schedule(static)
        for (int k = rank; k < n; k += world_size) {
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

    // 结果收集 (这里为了简化，只在rank 0上保留最终结果)
    // 实际应用中可能需要更复杂的MPI_Gather操作
    if (rank != 0) {
      for(int k=rank; k < n; k+=world_size) {
          MPI_Send(m[k].data(), n + 1, MPI_FLOAT, 0, k, MPI_COMM_WORLD);
      }
    } else {
      for (int source_rank = 1; source_rank < world_size; ++source_rank) {
          for (int k = source_rank; k < n; k += world_size) {
              MPI_Recv(m[k].data(), n + 1, MPI_FLOAT, source_rank, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
      }
    }
}


#endif // MPI_H