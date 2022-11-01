#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"
#include "cl/prefix_sum_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <climits>


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

    unsigned int workGroupSize = 128;
    unsigned int groups = (n + workGroupSize - 1) / workGroupSize;
    unsigned int batchSize = 4;
    unsigned int batchCount = 1 << batchSize;
    unsigned int histSize = groups * batchCount;

    gpu::gpu_mem_32u as_gpu, hist_gpu, hist_pref_gpu, hist_buf_gpu, bs_gpu;
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);
    hist_gpu.resizeN(histSize);
    hist_pref_gpu.resizeN(histSize);
    hist_buf_gpu.resizeN(histSize);

    {

        std::string defines = " -D WORK_GROUP_SIZE=" + to_string(workGroupSize);
        defines += " -D BATCH_SIZE=" + to_string(batchSize);

        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix", defines);
        radix.compile();

        ocl::Kernel count(radix_kernel, radix_kernel_length, "count", defines);
        count.compile();

        ocl::Kernel init(prefix_sum_kernel, prefix_sum_kernel_length, "init");
        init.compile();

        ocl::Kernel update(prefix_sum_kernel, prefix_sum_kernel_length, "update");
        update.compile();

        ocl::Kernel reduce(prefix_sum_kernel, prefix_sum_kernel_length, "reduce");
        reduce.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            unsigned int global_work_size = groups * workGroupSize;

            for (unsigned int bit = 0; bit < sizeof(unsigned int) * CHAR_BIT; bit += batchSize) {
                count.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, hist_gpu, n, bit);

                { // pref sum
                    unsigned int fullSize = (histSize + workGroupSize - 1) / workGroupSize * workGroupSize;
                    init.exec(gpu::WorkSize(workGroupSize, fullSize), hist_pref_gpu, histSize);
                    unsigned int m = histSize;
                    auto reduceWorkSize = [&]() {
                        return ((m + 1) / 2 + workGroupSize - 1) / workGroupSize * workGroupSize;
                    };
                    for (unsigned int pref_bit = 0; (1 << pref_bit) <= histSize; ++pref_bit) {
                        update.exec(gpu::WorkSize(workGroupSize, fullSize), hist_gpu, hist_pref_gpu, histSize, pref_bit);
                        reduce.exec(gpu::WorkSize(workGroupSize, reduceWorkSize()), hist_gpu, hist_buf_gpu, m);
                        m = (m + 1) / 2;
                        std::swap(hist_gpu, hist_buf_gpu);
                    }
                }

                radix.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, bs_gpu, n, hist_pref_gpu, bit);
                std::swap(as_gpu, bs_gpu);
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
