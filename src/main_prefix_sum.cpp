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
    // chooseGPUDevice:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    // Этот контекст после активации будет прозрачно использоваться при всех вызовах в libgpu библиотеке
    // это достигается использованием thread-local переменных, т.е. на самом деле контекст будет активирован для текущего потока исполнения
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

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
            bs.assign(n, 0);
            gpu::gpu_mem_32u as_gpu, bs_gpu, cs_gpu;
            as_gpu.resizeN(n);
            bs_gpu.resizeN(n);
            cs_gpu.resizeN(n);

            ocl::Kernel prefix_sum_bin(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum");
            prefix_sum_bin.compile();
            ocl::Kernel prefix_sum_other(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum_other");
            prefix_sum_other.compile();

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);
                bs_gpu.writeN(bs.data(), n);
                t.restart();

                for (unsigned int level = 0; (1<<level) <= n; level++) {
                    prefix_sum_bin.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                        as_gpu, bs_gpu, n, level);

                    unsigned int global_work_size2 = (n / (1<<(level+1)) + workGroupSize - 1) / workGroupSize * workGroupSize;
                    if (global_work_size2 > 0) {
                        prefix_sum_other.exec(gpu::WorkSize(workGroupSize, global_work_size2),
                                              as_gpu, cs_gpu, n / (1 << (level + 1)));
                    }
                    as_gpu.swap(cs_gpu);
                }

                t.nextLap();
            }

            bs_gpu.readN(bs.data(), n);

            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(bs[i], reference_result[i], "GPU results should be equal to CPU results!");
            }

            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

		}
	}
}
