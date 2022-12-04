#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/prefix_sum_cl.h"


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
	unsigned int max_n = (1 << 24);

	for (unsigned int n = 2; n <= max_n; n *= 2) {
		std::cout << "______________________________________________" << std::endl;
		unsigned int values_range = std::min<unsigned int>(1023, std::numeric_limits<int>::max() / n);
		std::cout << "n=" << n << " values in range: [" << 0 << "; " << values_range << "]" << std::endl;

		std::vector<unsigned int> as(n, 0);
		FastRandom r(n);
		for (int i = 0; i < n; ++i) {
			as[i] = r.next(0, values_range);
		}

		std::vector<unsigned int> bs(n, 0);
		{
			for (int i = 0; i < n; ++i) {
				bs[i] = as[i];
				if (i) {
					bs[i] += bs[i-1];
				}
			}
		}
		const std::vector<unsigned int> reference_result = bs;

		{
			{
				std::vector<unsigned int> result(n);
				for (int i = 0; i < n; ++i) {
					result[i] = as[i];
					if (i) {
						result[i] += result[i-1];
					}
				}
				for (int i = 0; i < n; ++i) {
					EXPECT_THE_SAME(reference_result[i], result[i], "CPU result should be consistent!");
				}
			}

			std::vector<unsigned int> result(n);
			timer t;
			for (int iter = 0; iter < benchmarkingIters; ++iter) {
				for (int i = 0; i < n; ++i) {
					result[i] = as[i];
					if (i) {
						result[i] += result[i-1];
					}
				}
				t.nextLap();
			}
			std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
			std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
		}

		{
            std::vector<uint> result(n);
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);

            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();

            gpu::gpu_mem_32u as_gpu, bs_gpu, cs_gpu;
            as_gpu.resizeN(n);
            bs_gpu.resizeN(n);
            cs_gpu.resizeN(n);

            ocl::Kernel prefix_sum(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum");
            ocl::Kernel reduce_sum(prefix_sum_kernel, prefix_sum_kernel_length, "reduce_sum");

            prefix_sum.compile();
            reduce_sum.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);
                bs_gpu.writeN(std::vector<uint>(n, 0).data(), n);

                t.restart();
                unsigned int workGroupSize = 256;
                unsigned int prefix_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
                unsigned int reduce_work_size = (n / 2 + workGroupSize - 1) / workGroupSize * workGroupSize;

                for (unsigned int bit = 0; (1 << bit) <= n; bit++) {
                    prefix_sum.exec(gpu::WorkSize(workGroupSize, prefix_work_size), bs_gpu, as_gpu, bit, n);
                    reduce_sum.exec(gpu::WorkSize(workGroupSize, reduce_work_size), as_gpu, cs_gpu, n >> (bit + 1));
                    std::swap(as_gpu, cs_gpu);
                }
                t.nextLap();

                bs_gpu.readN(result.data(), n);

                for (int i = 0; i < n; ++i) {
                    EXPECT_THE_SAME(reference_result[i], result[i], "GPU results should be equal to CPU results!");
                }
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}