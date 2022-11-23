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
    unsigned int n = 32 * 1024 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
//        as[i] = float(i);
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
            unsigned int workGroupSize = 1;
            unsigned int global_work_size = n/2;
            merge.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, bs_gpu, n);
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
        as_gpu.readN(as.data(), n);
    }
    // Проверяем корректность результатов
//    std::cout.precision(9);
//    for (int i = 155; i < 155+20; ++i) {
//        std::cout << as[i] << " ";
//    }
//    std::cout << std::endl;
//    for (int i = 0; i < 20; ++i) {
//        std::cout << as[i] << " ";
//    }
//    std::cout << std::endl;
//
//    std::vector<std::pair<int,float>> errors;
//
//    for (int i = 0; i < n; i++) {
//        if (as[i] < -999.4)
//            errors.push_back({i, as[i]});
//    }
//
//    std::cout << std::endl;
//    for (int i = 0; i < 10; i++) {
//        std::cout << errors[i].first << ":" << errors[i].second << " ";
//    }
//    std::cout << std::endl;
//
//    std::cout << std::endl;
//    for (int i = 0; i < 10; i++) {
//        if (as[i] == -999.913696)
//        std::cout << i << " ";
//    }
//    std::cout << std::endl;

    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
