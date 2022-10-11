#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_multiplication_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>

int benchmarkingIters = 10;

unsigned int M = 1024;
unsigned int K = 1024;
unsigned int N = 1024;
const size_t gflops = (((unsigned long long) M) * K * N * 2 / 1000 / 1000 / 1000); // умножить на два, т.к. операция сложения и умножения

template<class Num>
Num div_up(Num num, Num denom) {
  return (num + denom - 1) / denom;
}

template<class Num>
Num round_up(Num num, Num denom) {
  return div_up(num, denom) * denom;
}

int run_kernel(std::string kernel_name,
               std::vector<int>& as, std::vector<int>& bs, std::vector<int>& cs,
               const std::vector<int>& cs_cpu_reference,
               unsigned int x_items_per_thread)
{
  gpu::gpu_mem_32i as_gpu, bs_gpu, cs_gpu;
  as_gpu.resizeN(M * K);
  bs_gpu.resizeN(K * N);
  cs_gpu.resizeN(M * N);

  as_gpu.writeN(as.data(), M * K);
  bs_gpu.writeN(bs.data(), K * N);

  ocl::Kernel matrix_multiplication_kernel(matrix_multiplication,
                                           matrix_multiplication_length,
                                           kernel_name);
  matrix_multiplication_kernel.compile();

  {
    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
      unsigned int work_group_size_x = 16 / x_items_per_thread;
      unsigned int work_group_size_y = 16;
      unsigned int global_work_size_x = round_up(div_up(M, x_items_per_thread), work_group_size_x);
      unsigned int global_work_size_y = round_up(N, work_group_size_y);
      matrix_multiplication_kernel.exec(
          gpu::WorkSize(work_group_size_x, work_group_size_y, global_work_size_x, global_work_size_y),
          as_gpu, bs_gpu, cs_gpu, M, K, N);

      t.nextLap();
    }
    std::cout << "GPU (" << kernel_name << "):" << std::endl;
    std::cout << "  " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "  " << gflops / t.lapAvg() << " GFlops" << std::endl;
  }

  cs_gpu.readN(cs.data(), M * N);

  {
    // Проверяем корректность результатов
    double diff_sum = 0;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        double a = cs[i * N + j];
        double b = cs_cpu_reference[i * N + j];
        if (std::max(fabs(a), fabs(b)) != 0) {
          double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
          diff_sum += diff;
          if (diff > 0.01) {
            std::cerr << "Too big difference (" << diff << ")! (i=" << i << ", j=" << j << ")" << std::endl;
            return 1;
          }
        }
      }
    }

    double diff_avg = diff_sum / (M * N);
    std::cout << "Average difference: " << diff_avg * 100.0 << "%" << std::endl;
    if (diff_avg > 0.01) {
      std::cerr << "Too big difference (" << diff_avg << ")!" << std::endl;
      return 1;
    }
  }

  return 0;
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    std::vector<int> as(M*K, 0);
    std::vector<int> bs(K*N, 0);
    std::vector<int> cs(M*N, 0);

    FastRandom r(M+K+N);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.next() % 500;
    }
    for (unsigned int i = 0; i < bs.size(); ++i) {
        bs[i] = r.next() % 500;
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << ", N=" << N << "!" << std::endl;

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            for (int j = 0; j < M; ++j) {
                for (int i = 0; i < N; ++i) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += as.data()[j * K + k] * bs.data()[k * N + i];
                    }
                    cs.data()[j * N + i] = sum;
                }
            }
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    const std::vector<int> cs_cpu_reference = cs;

    if (run_kernel("matrix_multiplication_chunked", as, bs, cs, cs_cpu_reference, 1)) {
      return 1;
    }
    if (run_kernel("matrix_multiplication_multijob", as, bs, cs, cs_cpu_reference, 4)) {
      return 1;
    }

    return 0;
}
