#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/merge_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>


template<typename T>
void raiseFail(const T &a, const T &b, unsigned int i, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << " (i=" << i << "), " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, i, message) raiseFail(a, b, i, message, __FILE__, __LINE__)

template<class Num>
Num div_up(Num num, Num denom) {
  return (num + denom - 1) / denom;
}

template<class Num>
Num round_up(Num num, Num denom) {
  return div_up(num, denom) * denom;
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 1;
    unsigned int n = 3 * 2 * 4 /*128*/ * 2;//32 * 1024 * 1024; // todo
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = ((int) r.nextf() % 5) + 5;
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<float> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);
    {
        ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge_sort");
        merge.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфера данных
            unsigned int workGroupSize = 4/*128*/;
            unsigned int locallySortable = 3;
            unsigned int global_work_size = round_up(div_up(n, 2 * locallySortable), workGroupSize);
            merge.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, n);
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
        as_gpu.readN(as.data(), n);
    }
    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], i, "GPU results should be equal to CPU results!");
    }

    return 0;
}
