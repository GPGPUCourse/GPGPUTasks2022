#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include "libgpu/context.h"

#include "cl/sum_cl.h"
#include "libgpu/shared_device_buffer.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


void gpu_running(const std::string& kernel_name,
                 unsigned int n,
                 int benchmarkingIters,
                 unsigned int reference_sum,
                 std::vector<unsigned int> &as) {
    const bool printLog = false;
    const unsigned int workGroupSize = 128;

    ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_name); // choose name of sum method

    kernel.compile(printLog);

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);
    as_gpu.writeN(as.data(), n);

    unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
    if (kernel_name == "iter_sum" || kernel_name == "coalesced_sum") {
        global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize / 64;
    }

    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        std::vector<unsigned int> result(1, 0);

        gpu::gpu_mem_32u result_gpu;
        result_gpu.resizeN(1);
        result_gpu.writeN(result.data(), 1);

        kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                    as_gpu, n, result_gpu);

        result_gpu.readN(result.data(), 1);

        EXPECT_THE_SAME(reference_sum, result[0], "GPU result should be consistent!");
        t.nextLap();
    }
    std::cout << "GPU " << kernel_name << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "GPU " << kernel_name << ": " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
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

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    {
        gpu_running("atomic_sum", n, benchmarkingIters, reference_sum, as);
        gpu_running("iter_sum", n, benchmarkingIters, reference_sum, as);
        gpu_running("coalesced_sum", n, benchmarkingIters, reference_sum, as);
        gpu_running("local_sum", n, benchmarkingIters, reference_sum, as);
        gpu_running("tree_sum", n, benchmarkingIters, reference_sum, as);
    }
}
