#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cstdlib>
#include <mpi.h>

#include "baseline.h"
#include "simd.h"
#include "openmp.h"
#include "mpi.h"
#include "cuda.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::ofstream csv_file;
        csv_file.open("performance_results.csv");
        csv_file << "Matrix Size,"
                 << "Baseline Time (ms),"
                 << "SIMD Time (ms),"
                 << "SIMD+OMP Static Time (ms),"
                 << "SIMD+OMP Dynamic Time (ms),"
                 << "SIMD+OMP+MPI Time (ms),"
                 << "GPU Time (ms),"
                 << "Speedup SIMD,"
                 << "Speedup OMP Static,"
                 << "Speedup OMP Dynamic,"
                 << "Speedup MPI,"
                 << "Speedup GPU\n";

        for (int size = 128; size <= 4096; size += 128) {
            std::cout << "Testing size: " << size << "..." << std::endl;
            
            matrix m_base, m_simd, m_omp_s, m_omp_d, m_mpi, m_gpu;
            init_matrix(m_base, size);
            m_simd = m_omp_s = m_omp_d = m_mpi = m_gpu = m_base; // 确保每次测试的初始矩阵相同

            // --- Baseline ---
            auto start = std::chrono::high_resolution_clock::now();
            gaussian_elimination_baseline(m_base);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> baseline_time = end - start;
            
            // --- SIMD ---
            start = std::chrono::high_resolution_clock::now();
            gaussian_elimination_simd(m_simd);
            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> simd_time = end - start;

            // --- SIMD + OpenMP Static ---
            start = std::chrono::high_resolution_clock::now();
            gaussian_elimination_omp_static(m_omp_s);
            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> omp_s_time = end - start;

            // --- SIMD + OpenMP Dynamic ---
            start = std::chrono::high_resolution_clock::now();
            gaussian_elimination_omp_dynamic(m_omp_d);
            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> omp_d_time = end - start;

            // --- SIMD + OpenMP + MPI ---
            // MPI部分的时间测量包含通信，所以需要特殊处理
            MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD); // 通知其他进程矩阵大小
            auto mpi_start = MPI_Wtime();
            gaussian_elimination_mpi(m_mpi, rank, world_size);
            auto mpi_end = MPI_Wtime();
            double mpi_time_sec = mpi_end - mpi_start;


            // --- GPU ---
            start = std::chrono::high_resolution_clock::now();
            gaussian_elimination_cuda_wrapper(m_gpu);
            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> gpu_time = end - start;

            // --- Write to CSV ---
            csv_file << size << ","
                     << baseline_time.count() << ","
                     << simd_time.count() << ","
                     << omp_s_time.count() << ","
                     << omp_d_time.count() << ","
                     << mpi_time_sec * 1000.0 << "," // 转为毫秒
                     << gpu_time.count() << ","
                     << baseline_time.count() / simd_time.count() << ","
                     << baseline_time.count() / omp_s_time.count() << ","
                     << baseline_time.count() / omp_d_time.count() << ","
                     << baseline_time.count() / (mpi_time_sec * 1000.0) << ","
                     << baseline_time.count() / gpu_time.count() << "\n";
        }
        
        int stop_signal = 0;
        MPI_Bcast(&stop_signal, 1, MPI_INT, 0, MPI_COMM_WORLD); // 发送结束信号

        csv_file.close();
        std::cout << "Testing finished. Results are in performance_results.csv" << std::endl;

    } else { // Worker aroc
        while (true) {
            int size;
            MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (size == 0) break; // 收到结束信号
            
            matrix m(size, std::vector<float>(size + 1));
            // worker也需要跑MPI版本
            gaussian_elimination_mpi(m, rank, world_size);
        }
    }

    MPI_Finalize();
    return 0;
}