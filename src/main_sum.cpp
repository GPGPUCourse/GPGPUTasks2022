#include <libgpu/shared_device_buffer.h>
#include <libgpu/context.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/sum_cl.h"

#include <map>
#include <string>

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


gpu::gpu_mem_32u runRecursiveTree(ocl::Kernel kernel, gpu::gpu_mem_32u a, gpu::gpu_mem_32u b, unsigned int n);

int main(int argc, char **argv)
{
    int benchmarkingIters = 10;
    int cpuBenchmarkingIters = 5;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000 + 17;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    std::cout << "Reference sum: " << reference_sum << std::endl;

    {
        timer t;
        for (int iter = 0; iter < cpuBenchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
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
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);
    as_gpu.writeN(as.data(), n);

    gpu::gpu_mem_32u res_gpu;
    res_gpu.resizeN(1);

    const size_t defGroupSize = 128;

    std::map<std::string, gpu::WorkSize> kernels;
    kernels.emplace("sumGlobalAdd", gpu::WorkSize(defGroupSize, (n + defGroupSize - 1) / defGroupSize * defGroupSize));
    kernels.emplace("sumLoop", gpu::WorkSize(defGroupSize, (n + defGroupSize - 1) / defGroupSize * defGroupSize / 64));
    kernels.emplace("sumLoopCoalesced", gpu::WorkSize(defGroupSize, (n + defGroupSize - 1) / defGroupSize * defGroupSize / 64));
    kernels.emplace("sumMajorWorker", gpu::WorkSize(defGroupSize, (n + defGroupSize - 1) / defGroupSize * defGroupSize));
    kernels.emplace("sumTree", gpu::WorkSize(defGroupSize, (n + defGroupSize - 1) / defGroupSize * defGroupSize));

    for (const auto& entry : kernels) {
        const std::string& kernelName = entry.first;
        gpu::WorkSize workSize = entry.second;

        unsigned int res = 0;
        res_gpu.writeN(&res, 1);

        ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernelName);
//        std::cout << "Preparing kernel " << kernelName << std::endl;
        kernel.compile(false);

        timer t;
        for (int i = 0; i < benchmarkingIters; ++i) {
            kernel.exec(workSize, as_gpu, n, res_gpu);
            if (i == 0) {
                res_gpu.readN(&res, 1);
            }
            t.nextLap();
        }
        std::cout << "GPU " << kernelName << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU " << kernelName << ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;

        EXPECT_THE_SAME(reference_sum, res, "GPU result should be consistent!");
    }

    // recursive tree
    {
        const std::string& kernelName = "sumTreeRecursive";

        unsigned int res = 0;
        res_gpu.writeN(&res, 1);

        ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernelName);
//        std::cout << "Preparing kernel " << kernelName << std::endl;
        kernel.compile(false);

        gpu::gpu_mem_32u bs_gpu;
        bs_gpu.resizeN(n);

        timer t;
        for (int i = 0; i < benchmarkingIters; ++i) {
            gpu::gpu_mem_32u rtRes_gpu = runRecursiveTree(kernel, as_gpu, bs_gpu, n);
            if (i == 0) {
                rtRes_gpu.readN(&res, 1);
            }
            t.nextLap();
        }
        std::cout << "GPU " << kernelName << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU " << kernelName << ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;

        EXPECT_THE_SAME(reference_sum, res, "GPU result should be consistent!");
    }
}

gpu::gpu_mem_32u runRecursiveTree(ocl::Kernel kernel, gpu::gpu_mem_32u a, gpu::gpu_mem_32u b, unsigned int n) {
    const unsigned int gs = 128;
    while (n > 1) {
        kernel.exec(gpu::WorkSize(gs, (n + gs - 1) / gs * gs), a, n, b);
        n = (n + gs - 1) / gs;
        std::swap(a, b);
    }
    return a;
}
