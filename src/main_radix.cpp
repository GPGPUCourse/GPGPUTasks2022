#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
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

    gpu::gpu_mem_32u as_gpu, zeros_gpu, ones_gpu, prefix_zeros_gpu, prefix_ones_gpu, tmp_gpu, last_gpu;
    as_gpu.resizeN(n);
    zeros_gpu.resizeN(n);
    ones_gpu.resizeN(n);
    prefix_zeros_gpu.resizeN(n);
    prefix_ones_gpu.resizeN(n);
    tmp_gpu.resizeN(n);
    last_gpu.resizeN(1);

    std::vector<unsigned int> zeros(n, 0);

    {
        ocl::Kernel get_zeros_and_ones(radix_kernel, radix_kernel_length, "get_zeros_and_ones");
        get_zeros_and_ones.compile();
        ocl::Kernel prefix_sum(radix_kernel, radix_kernel_length, "prefix_sum");
        prefix_sum.compile();
        ocl::Kernel pairwise_sum(radix_kernel, radix_kernel_length, "pairwise_sum");
        pairwise_sum.compile();
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            gpu::WorkSize ws = gpu::WorkSize(workGroupSize, global_work_size);

            for (unsigned int digit = 0; digit < 32; digit++) {
                get_zeros_and_ones.exec(ws, as_gpu, n, digit, zeros_gpu, ones_gpu);

                // Составляем частичные суммы для нулей
                prefix_zeros_gpu.writeN(zeros.data(), n);
                unsigned int m = n / 2, shift = 0;
                while (true) {
                    prefix_sum.exec(ws, zeros_gpu, prefix_zeros_gpu, n, shift, last_gpu, 1);
                    if (m == 0) break;
                    pairwise_sum.exec(ws, zeros_gpu, tmp_gpu, m);
                    zeros_gpu.swap(tmp_gpu);
                    m /= 2;
                    shift++;
                }

                // Составляем частичные суммы для единиц
                prefix_ones_gpu.writeN(zeros.data(), n);
                m = n / 2;
                shift = 0;
                while (true) {
                    prefix_sum.exec(ws, ones_gpu, prefix_ones_gpu, n, shift, last_gpu, 0);
                    if (m == 0) break;
                    pairwise_sum.exec(ws, ones_gpu, tmp_gpu, m);
                    ones_gpu.swap(tmp_gpu);
                    m /= 2;
                    shift++;
                }

                // Делаем шаг сортировки
                radix.exec(ws, as_gpu, tmp_gpu, prefix_zeros_gpu, prefix_ones_gpu, n, digit, last_gpu);
                as_gpu.swap(tmp_gpu);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
