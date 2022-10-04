#include "cl/sum_cl.h"
#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include <utility>

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void bench(std::string algoName,
           std::vector<unsigned int> &data,
           unsigned int trueSum,
           int benchmarkingIters,
           unsigned int workGroupSize,
           unsigned int globalWorkSize)
{
   ocl::Kernel kernel(sum_kernel, sum_kernel_length, std::move(algoName));
   bool printLog = false;
   kernel.compile(printLog);

   gpu::gpu_mem_32u result;
   result.resizeN(1);

   gpu::gpu_mem_32u gpu_data;
   gpu_data.resizeN(data.size());
   gpu_data.writeN(data.data(), data.size());

   timer t;
   for (int i = 0; i < benchmarkingIters; ++i) {
       unsigned int sum = 0;
       result.writeN(&sum, 1);
       kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                   gpu_data,
                   (unsigned int) data.size(),
                   result);
       result.readN(&sum, 1);
       EXPECT_THE_SAME(trueSum, sum, "GPU result should be consistent!");
       t.nextLap();
   }

   std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
   std::cout << "GPU:     " << (data.size()/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
}

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
        // implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        const unsigned int WORK_GROUP_SIZE = 256;
        const unsigned int DATA_PER_ITEM = 64;
        const std::map<std::string, unsigned int> algoToGlobalWorkSize {
            {"sum_1", n},
            {"sum_2", n / DATA_PER_ITEM},
            {"sum_3", n / DATA_PER_ITEM},
            {"sum_4", n},
            {"sum_5", n}
        };

        for (const auto& item : algoToGlobalWorkSize) {
          auto algoName = item.first;
          auto globalWorkSize = item.second;

          std::cout << algoName << std::endl;
          bench(algoName, as, reference_sum, benchmarkingIters, WORK_GROUP_SIZE, globalWorkSize);
        }
    }
}
