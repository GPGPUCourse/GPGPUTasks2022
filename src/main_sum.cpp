#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"

#include "cl/sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

unsigned int workGroupSize = 128;
void count_sums(const std::string& kernel_name, gpu::Device &device, unsigned int n, std::vector<unsigned int> as, int benchmarkingIters,
                unsigned int reference_sum, unsigned int globalWorkSize) {
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();
    {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_name);
        bool printLog = false;
        kernel.compile(printLog);

        gpu::gpu_mem_32u gpu_input, gpu_result;
        gpu_result.resizeN(1);
        gpu_input.resizeN(n);
        gpu_input.writeN(as.data(), n);

        timer t;


        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int res = 0;
            gpu_result.writeN(&res, 1);
            kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                        gpu_input, n, gpu_result);
            gpu_result.readN(&res, 1);
            EXPECT_THE_SAME(res, reference_sum, "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << kernel_name << " GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << kernel_name << " GPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}

int main(int argc, char **argv) {
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
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
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
#pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        unsigned int globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        count_sums("sum_gpu_1", device, n, as, benchmarkingIters, reference_sum, globalWorkSize);
        count_sums("sum_gpu_2", device, n, as, benchmarkingIters, reference_sum, globalWorkSize / 64);
        count_sums("sum_gpu_3", device, n, as, benchmarkingIters, reference_sum, globalWorkSize / 64);
        count_sums("sum_gpu_4", device, n, as, benchmarkingIters, reference_sum, globalWorkSize);
        count_sums("sum_gpu_5", device, n, as, benchmarkingIters, reference_sum, globalWorkSize);
    }
}
