#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_multiplication_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    int cpuBenchmarkingIters = 5;
    unsigned int M = 1024;
    unsigned int K = 512;
    unsigned int N = 2 * 1024;
    const size_t gflop = ((size_t) M * K * N * 2) / (1000 * 1000 * 1000); // умножить на два, т.к. операция сложения и умножения

    std::vector<float> as(M*K, 0);
    std::vector<float> bs(K*N, 0);
    std::vector<float> cs(M*N, 0);
    std::vector<float> cs2(M*N, 0);

    FastRandom r(M+K+N);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    for (unsigned int i = 0; i < bs.size(); ++i) {
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << ", N=" << N << "!" << std::endl;

    {
        timer t;
        for (int iter = 0; iter < cpuBenchmarkingIters; ++iter) {
            for (int j = 0; j < M; ++j) {
                for (int i = 0; i < N; ++i) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += as[j * K + k] * bs[k * N + i];
                    }
                    cs[j * N + i] = sum;
                }
            }
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << gflop / t.lapAvg() << " GFlops" << std::endl;
    }

    const std::vector<float> cs_cpu_reference = cs;

    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(M*K);
    bs_gpu.resizeN(K*N);
    cs_gpu.resizeN(M*N);

    as_gpu.writeN(as.data(), M*K);
    bs_gpu.writeN(bs.data(), K*N);

    ocl::Kernel matrix_multiplication_kernel(matrix_multiplication, matrix_multiplication_length, "matrix_multiplication");
    matrix_multiplication_kernel.compile();

    {
      timer t;
      for (int iter = 0; iter < benchmarkingIters; ++iter) {
        matrix_multiplication_kernel.exec(gpu::WorkSize(16, 16, N, M), as_gpu, bs_gpu, cs_gpu, M, K, N);

        t.nextLap();
      }
      std::cout << "First kernel:\n";
      std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
      std::cout << "GPU: " << gflop / t.lapAvg() << " GFlops\n" << std::endl;
    }

    cs_gpu.readN(cs.data(), M*N);

    ocl::Kernel matrix_multiplication2_kernel(matrix_multiplication, matrix_multiplication_length, "matrix_multiplication2");
    matrix_multiplication2_kernel.compile();
    const unsigned int tileSize = 32;
    const unsigned int stripeSize = 8;

    {
      timer t;
      for (int iter = 0; iter < benchmarkingIters; ++iter) {
        matrix_multiplication2_kernel.exec(gpu::WorkSize(tileSize, tileSize / stripeSize, N, M / stripeSize), as_gpu, bs_gpu, cs_gpu, M, K, N);

        t.nextLap();
      }
      std::cout << "Second kernel:\n";
      std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
      std::cout << "GPU: " << gflop / t.lapAvg() << " GFlops\n" << std::endl;
    }

    cs_gpu.readN(cs2.data(), M*N);

    // Проверяем корректность результатов
    double diff_sum = 0;
    double diff2_sum = 0;
    for (int i = 0; i < M * N; ++i) {
        double a = cs[i];
        double a2 = cs2[i];
        double b = cs_cpu_reference[i];
        if (a != 0.0 && b != 0.0) {
          double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
          diff_sum += diff;

          diff = fabs(a2 - b) / std::max(fabs(a2), fabs(b));
          diff2_sum += diff;
        }
    }

    std::cout << "First kernel:\n";
    double diff_avg = diff_sum / (M * N);
    std::cout << "Average difference: " << diff_avg * 100.0 << "%" << std::endl;
    if (diff_avg > 0.01) {
        std::cerr << "Too big difference!" << std::endl;
        return 1;
    }

    std::cout << "Second kernel:\n";
    double diff2_avg = diff2_sum / (M * N);
    std::cout << "Average difference: " << diff2_avg * 100.0 << "%" << std::endl;
    if (diff2_avg > 0.01) {
      std::cerr << "Too big difference!" << std::endl;
      return 1;
    }

    return 0;
}
