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

#define VALUES_PER_WORK_ITEM 64

int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
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

    {
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u as_gpu, sum_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);

        {
            unsigned int sum = 0;
            sum_gpu.resizeN(1);
            sum_gpu.writeN(&sum, 1);

            ocl::Kernel sum_ker(sum_kernel, sum_kernel_length, "sum_1");
            sum_ker.compile();

            unsigned int workGroupSize = 256;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                sum_gpu.writeN(&sum, 1);
                sum_ker.exec(gpu::WorkSize(1, global_work_size),
                             as_gpu, sum_gpu, n);
                t.nextLap();
            }

            sum_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU baseline result should be consistent!");

            std::cout << "GPU baseline: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU baseline: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            unsigned int sum = 0;
            sum_gpu.resizeN(1);
            sum_gpu.writeN(&sum, 1);

            ocl::Kernel sum_ker(sum_kernel, sum_kernel_length, "sum_2");
            sum_ker.compile();

            unsigned int workGroupSize = 256;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                sum_gpu.writeN(&sum, 1);
                sum_ker.exec(gpu::WorkSize(1, global_work_size),
                             as_gpu, sum_gpu, n);
                t.nextLap();
            }

            sum_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU cycle result should be consistent!");

            std::cout << "GPU cycle: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU cycle: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            unsigned int sum = 0;
            sum_gpu.resizeN(1);
            sum_gpu.writeN(&sum, 1);

            ocl::Kernel sum_ker(sum_kernel, sum_kernel_length, "sum_3");
            sum_ker.compile();

            unsigned int workGroupSize = 64;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize;

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                sum_gpu.writeN(&sum, 1);
                sum_ker.exec(gpu::WorkSize(workGroupSize, global_work_size),
                             as_gpu, sum_gpu, n);
                t.nextLap();
            }

            sum_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU cycle with coalesced result should be consistent!");

            std::cout << "GPU cycle with coalesced: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU cycle with coalesced: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            unsigned int sum = 0;
            sum_gpu.resizeN(1);
            sum_gpu.writeN(&sum, 1);

            ocl::Kernel sum_ker(sum_kernel, sum_kernel_length, "sum_4");
            sum_ker.compile();

            unsigned int workGroupSize = 256;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                sum_gpu.writeN(&sum, 1);
                sum_ker.exec(gpu::WorkSize(workGroupSize, global_work_size),
                             as_gpu, sum_gpu, n);
                t.nextLap();
            }

            sum_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU local/global memory result should be consistent!");

            std::cout << "GPU local/global memory: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU local/global memory: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            unsigned int sum = 0;
            sum_gpu.resizeN(1);
            sum_gpu.writeN(&sum, 1);

            ocl::Kernel sum_ker(sum_kernel, sum_kernel_length, "sum_5");
            sum_ker.compile();

            unsigned int workGroupSize = 256;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                sum_gpu.writeN(&sum, 1);
                sum_ker.exec(gpu::WorkSize(workGroupSize, global_work_size),
                             as_gpu, sum_gpu, n);
                t.nextLap();
            }

            sum_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU tree result should be consistent!");

            std::cout << "GPU tree: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU tree: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }


}
