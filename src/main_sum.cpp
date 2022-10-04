#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

unsigned int roundup(unsigned int num, unsigned int denom) {
  return gpu::divup(num, denom) * denom;
}


int main(int argc, char **argv) {
  gpu::Device device = gpu::chooseGPUDevice(argc, argv);

  int benchmarkingIters = 10;

  unsigned int reference_sum = 0;
  unsigned int n = 100 * 1000 * 1000;
  std::vector<unsigned int> as(n, 0);
  FastRandom r(42);
  for (int i = 0; i < n; ++i) {
    as[i] =
        (unsigned int)r.next(0, std::numeric_limits<unsigned int>::max() / n);
    reference_sum += as[i];
  }

  {
    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
      unsigned int sum = 0;
      for (int i = 0; i < n; ++i) {
        sum += as[i];
      }
      EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
      t.nextLap();
    }
    std::cout << "CPU:" << std::endl;
    std::cout << "    " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "    " << (n / 1e9 / t.lapAvg()) << " GFlops" << std::endl;
  }

  {
    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
      unsigned int sum = 0;
#pragma omp parallel for reduction(+ : sum)
      for (int i = 0; i < n; ++i) {
        sum += as[i];
      }
      EXPECT_THE_SAME(reference_sum, sum,
                      "CPU OpenMP result should be consistent!");
      t.nextLap();
    }
    std::cout << "CPU OMP:" << std::endl;
    std::cout << "    " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "    " << (n / 1e9 / t.lapAvg()) << " GFlops" << std::endl;
  }

  {
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    unsigned int work_group_size = 128;
    unsigned int global_work_size_unbatched = roundup(n, work_group_size);

    unsigned int batch_size = 64;
    unsigned int global_work_size_batched = roundup(gpu::divup(n, batch_size), work_group_size);

    std::map<std::string, bool> fun_to_batched_map{
            {"atomic_sum", false},
            {"cycle_sum", true},
            {"coalesced_cycle_sum", true},
            {"local_sum", false},
            {"recursive_sum", false}
        };
    for (auto&fun_to_batched : fun_to_batched_map) {
      std::string kernel_name = fun_to_batched.first;
      unsigned int global_work_size = fun_to_batched.second ? global_work_size_batched : global_work_size_unbatched;
      ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_name);
      kernel.compile();

      gpu::gpu_mem_32u xs_gpu;
      gpu::gpu_mem_32u result_gpu;

      xs_gpu.resizeN(n);
      result_gpu.resizeN(1);

      xs_gpu.writeN(as.data(), n);

      unsigned int result;
      timer t;
      for (int iter = 0; iter < benchmarkingIters; ++iter) {
        result = 0;
        result_gpu.writeN(&result, 1);
        kernel.exec(gpu::WorkSize(work_group_size, global_work_size),
                    xs_gpu, n, batch_size, result_gpu);
        result_gpu.readN(&result, 1);
        EXPECT_THE_SAME(reference_sum, result,"GPU " + kernel_name + " result should be consistent!");
        t.nextLap();
      }

      std::cout << "GPU (" << kernel_name << "):" << std::endl;
      std::cout << "    " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
      std::cout << "    " << (n / 1e9 / t.lapAvg()) << " GFlops" << std::endl;
    }
  }
}
