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
            std::vector<unsigned int> array_of_zeros(n, 0);
            std::vector<unsigned int> result(n);

            gpu::Device device = gpu::chooseGPUDevice(argc, argv);

            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();


            gpu::gpu_mem_32u cur_prefix_sum_gpu;
            cur_prefix_sum_gpu.resizeN(n);

            gpu::gpu_mem_32u cur_part_sum_gpu;
            cur_part_sum_gpu.resizeN(n);
            gpu::gpu_mem_32u next_part_sum_gpu;
            next_part_sum_gpu.resizeN(n);
            
            ocl::Kernel prefix_sum(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum");
            prefix_sum.compile();
            ocl::Kernel part_sum(prefix_sum_kernel, prefix_sum_kernel_length, "part_sum");
            part_sum.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                cur_prefix_sum_gpu.writeN(array_of_zeros.data(), n);
                cur_part_sum_gpu.writeN(as.data(), n);

                t.restart();
                unsigned int work_group_size = 128;
                unsigned int prefix_sum_global_work_size = (n + work_group_size - 1) / work_group_size * work_group_size;
                unsigned int num_of_next_parts = n / 2;
                unsigned int part_sum_global_work_size = (num_of_next_parts + work_group_size - 1) / work_group_size * work_group_size;
                for (int cur_bit = 0; (1 << cur_bit) <= n; cur_bit++) {
                    prefix_sum.exec(gpu::WorkSize(work_group_size, prefix_sum_global_work_size),
                                    cur_prefix_sum_gpu, cur_part_sum_gpu, cur_bit, n);

                    part_sum.exec(gpu::WorkSize(work_group_size, part_sum_global_work_size),
                                  cur_part_sum_gpu, next_part_sum_gpu, num_of_next_parts);

                    cur_part_sum_gpu.swap(next_part_sum_gpu);
                    num_of_next_parts /= 2;
                }
                t.nextLap();

                cur_prefix_sum_gpu.readN(result.data(), n);
                for (int i = 0; i < n; ++i) {
                    EXPECT_THE_SAME(reference_result[i], result[i], "GPU result should be consistent!");
                }
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
		}
	}
}
