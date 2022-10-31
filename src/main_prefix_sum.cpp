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

template <class T>
std::ostream& operator <<(std::ostream& s, const std::vector<T>& data) {
    for (const T& x : data) {
        s << x << ' ';
    }
    return s;
}

int main(int argc, char **argv)
{
	int benchmarkingIters = 10;
	unsigned int max_n = (1 << 24);

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    gpu::gpu_mem_32u as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(max_n);
    bs_gpu.resizeN(max_n);
    cs_gpu.resizeN(max_n);

    ocl::Kernel kernelReduceAdjacent(prefix_sum_kernel, prefix_sum_kernel_length, "psum_reduce_adjacent");
    kernelReduceAdjacent.compile();
    ocl::Kernel kernelReduce(prefix_sum_kernel, prefix_sum_kernel_length, "psum_reduce");
    kernelReduce.compile();
    ocl::Kernel kernelFillZero(prefix_sum_kernel, prefix_sum_kernel_length, "psum_fill_zero");
    kernelFillZero.compile();

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
            size_t workGroupSize = 128;
            size_t workSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            gpu::WorkSize ws(workGroupSize, workSize);

            timer t;
            for (size_t i = 0; i < benchmarkingIters; ++i) {
                as_gpu.writeN(as.data(), n);

                t.restart();

                for (uint32_t step = 0; (1 << step) < 2 * n; ++step) {
                    if (step) {
                        uint32_t m = n >> (step - 1);
                        size_t reduceAdjacentWorkSize =
                                ((m + 1) / 2 + workGroupSize - 1) / workGroupSize * workGroupSize;
                        kernelReduceAdjacent.exec(gpu::WorkSize(workGroupSize, reduceAdjacentWorkSize),
                                                  as_gpu, cs_gpu, m);
                        std::swap(as_gpu, cs_gpu);
                    } else {
                        kernelFillZero.exec(ws, bs_gpu, n);
                    }
                    kernelReduce.exec(ws, as_gpu, bs_gpu, n, step);
                }

                t.nextLap();
            }

            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

            std::vector<unsigned int> result(n);
            bs_gpu.readN(result.data(), n);

            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(reference_result[i], result[i], "GPU result should be consistent!");
            }
		}
	}
}
