#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"

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
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        unsigned int valuesPerWorkItem = 64, workGroupSize = 128;
        for (int i = 1; i < 6; i++) {
            std::string name = "sum_gpu_" + std::to_string(i);
            unsigned int N;
            if ((i == 2) | (i == 3))
                N = ((n + valuesPerWorkItem - 1) / valuesPerWorkItem + workGroupSize - 1) / workGroupSize * workGroupSize;
            else
                N = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            ocl::Kernel kernel(sum_kernel, sum_kernel_length, name);
            kernel.compile();

            gpu::gpu_mem_32u as_gpu, res_gpu;
            as_gpu.resizeN(n);
            res_gpu.resizeN(1);

            as_gpu.writeN(as.data(), n);

            unsigned int res;
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                res = 0;
                res_gpu.writeN(&res, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, N),
                            as_gpu, n, res_gpu);
                res_gpu.readN(&res, 1);
                EXPECT_THE_SAME(reference_sum, res, "GPU " + name + " result should be consistent!");
                t.nextLap();
            }

            std::cout << name + " on GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << name + " on GPU: " << n / 1000.0 / 1000.0 / t.lapAvg() << " millions/s" << std::endl;

        }
    }
}