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

    int benchmarkingIters = 1;
    unsigned int n = 32 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
//        as[i] = float(n - i - 1);
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

//    for (int i = 0; i < 20; ++i) {
//        std::cout.precision(9);
//        std::cout << as[i] << " ";
//    }
//    std::cout << std::endl;

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

    gpu::gpu_mem_32f as_gpu, bs_gpu;
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);
    {
        ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge");
        merge.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфера данных
            for (int level_size = 2; level_size <= n; level_size*=2) {
                unsigned int workGroupSize = level_size/2;
                if (workGroupSize > 32)
                    workGroupSize = 32;
                unsigned int workGroupCount = n / level_size;
                merge.exec(gpu::WorkSize(workGroupSize, workGroupCount), as_gpu, n, level_size);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
        as_gpu.readN(as.data(), n);
    }
    // Проверяем корректность результатов

//    std::cout.precision(10);
//    std::cout << std::endl << "EXP: ";
//    for (int i = 0; i < 20; ++i) {
//        std::cout << cpu_sorted[i] << " ";
//    }
//    std::cout << std::endl;
//
//    std::cout << std::endl << "ACT: ";
//    for (int i = 0; i < 20; ++i) {
//        std::cout << as[i] << " ";
//    }
//    std::cout << std::endl;
//
//    std::cout << std::endl << "8192: ";
//    for (int i = 8192-5; i < 8192 + 20; i++) {
//        std::cout << as[i] << " ";
//    }
//    std::cout << std::endl;
//
//    std::cout << std::endl << "16384: ";
//    for (int i = 16384-5; i < 16384 + 20; i++) {
//        std::cout << as[i] << " ";
//    }
//    std::cout << std::endl;
//
//    std::cout << std::endl << "24576: ";
//    for (int i = 24576-5; i < 24576 + 20; i++) {
//        std::cout << as[i] << " ";
//    }
//    std::cout << std::endl;

//    std::cout << std::endl << "<'s: ";
//    for (int i = 1; i < n; i++) {
//        if (as[i-1] < as[i])
//            std::cout << i << " ";
//    }
//    std::cout << std::endl;

    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
