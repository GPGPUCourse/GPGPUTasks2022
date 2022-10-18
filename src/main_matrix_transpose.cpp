#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_transpose_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>

template<class Num>
Num round_up(Num num, Num denom) {
  return ((num + denom - 1) / denom) * denom;
}

int run_kernel(std::string kernel_name,
               unsigned int M, unsigned int K,
               std::vector<float>& as, std::vector<float>& as_t)
{
  ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, kernel_name);
  matrix_transpose_kernel.compile();

  gpu::gpu_mem_32f as_gpu, as_t_gpu;
  as_gpu.resizeN(M*K);
  as_t_gpu.resizeN(K*M);

  as_gpu.writeN(as.data(), M*K);

  {
    unsigned int work_group_size_x = 16;
    unsigned int work_group_size_y = 16;
    unsigned int global_work_size_x = round_up(M, work_group_size_x);
    unsigned int global_work_size_y = round_up(K, work_group_size_y);

    int benchmarkingIters = 10;
    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
      // Для этой задачи естественнее использовать двухмерный NDRange. Чтобы это сформулировать
      // в терминологии библиотеки - нужно вызвать другую вариацию конструктора WorkSize.
      // В CLion удобно смотреть какие есть вариант аргументов в конструкторах:
      // поставьте каретку редактирования кода внутри скобок конструктора WorkSize -> Ctrl+P -> заметьте что есть 2, 4 и 6 параметров
      // - для 1D, 2D и 3D рабочего пространства соответственно
      matrix_transpose_kernel.exec(
          gpu::WorkSize(work_group_size_x, work_group_size_y, global_work_size_x, global_work_size_y),
          as_gpu, as_t_gpu, M, K
      );

      t.nextLap();
    }
    std::cout << "  GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "  GPU: " << M*K / 1e9 / t.lapAvg() << " GFlops" << std::endl;
  }

  as_t_gpu.readN(as_t.data(), M*K);

  // Проверяем корректность результатов
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < K; ++col) {
      float a = as[row * K + col];
      float b = as_t[col * M + row];
      if (a != b) {
        std::cerr << "Not the same! " << std::endl;
        return 1;
      }
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

  unsigned int M = 8192;
  unsigned int K = 8192;

  std::vector<float> as(M*K, 0);
  std::vector<float> as_t(M*K, 0);

  FastRandom r(M+K);
  for (unsigned int i = 0; i < as.size(); ++i) {
    as[i] = r.nextf();
  }
  std::cout << "Data generated for M=" << M << ", K=" << K << "!" << std::endl;

  std::cout << "Regular transpose:" << std::endl;
  if (run_kernel("matrix_transpose", M, K, as, as_t)) {
    return 1;
  }
  std::cout << "Skewed transpose:" << std::endl;
  if (run_kernel("matrix_transpose_skewed", M, K, as, as_t)) {
    return 1;
  }

  return 0;
}
