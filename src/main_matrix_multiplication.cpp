#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_multiplication_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>
#include <iomanip>


bool check_result(unsigned int M, unsigned int N,
                  const std::vector<float> &cs, const std::vector<float> &cs_cpu_reference) {
    // Проверяем корректность результатов
    double diff_sum = 0;
    for (int i = 0; i < M * N; ++i) {
        double a = cs[i];
        double b = cs_cpu_reference[i];
        if (a != 0.0 && b != 0.0) {
            double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
            diff_sum += diff;
        }
    }

    double diff_avg = diff_sum / (M * N);
    std::cout << "Average difference: " << diff_avg * 100.0 << "%" << std::endl;
    if (diff_avg > 0.01) {
        std::cerr << "Too big difference!" << std::endl;
        return false;
    }
    return true;
}


void gpu_running(const std::string &kernel_name,
                 const gpu::gpu_mem_32f &as_gpu,
                 const gpu::gpu_mem_32f &bs_gpu,
                 const gpu::gpu_mem_32f &cs_gpu,
                 const gpu::WorkSize work_size,
                 unsigned int M, unsigned int N, unsigned int K,
                 int benchmarkingIters, size_t gflops,
                 std::vector<float> &cs) {
    ocl::Kernel matrix_multiplication_kernel(matrix_multiplication, matrix_multiplication_length, kernel_name);
    matrix_multiplication_kernel.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            matrix_multiplication_kernel.exec(work_size,as_gpu, bs_gpu, cs_gpu, M, K, N);

            t.nextLap();
        }
        std::cout << "GPU " << kernel_name << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU " << kernel_name << ": " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    cs_gpu.readN(cs.data(), M*N);
}


void print(unsigned int N, unsigned int M, std::vector<float> &cs) {
    for (int j = 0; j < M; j++) {
        for (int i = 0; i < N; i++) {
            std::cout << std::fixed << std::showpoint << std::setprecision(2) << cs[j * N + i] << " ";
        }
        std::cout << '\n';
    }
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 1;
    unsigned int M = 1024;
    unsigned int K = 1024;
    unsigned int N = 1024;
    const size_t gflops = ((size_t) M * K * N * 2) / (1000 * 1000 * 1000); // умножить на два, т.к. операция сложения и умножения

    std::vector<float> as(M*K, 0);
    std::vector<float> bs(K*N, 0);
    std::vector<float> cs(M*N, 0);

    FastRandom r(M+K+N);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    for (unsigned int i = 0; i < bs.size(); ++i) {
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << ", N=" << N << "!" << std::endl;

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            for (int j = 0; j < M; ++j) {
                for (int i = 0; i < N; ++i) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += as.data()[j * K + k] * bs.data()[k * N + i];
                    }
                    cs.data()[j * N + i] = sum;
                }
            }
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    const std::vector<float> cs_cpu_reference = cs;

    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(M*K);
    bs_gpu.resizeN(K*N);
    cs_gpu.resizeN(M*N);

    as_gpu.writeN(as.data(), M*K);
    bs_gpu.writeN(bs.data(), K*N);

    gpu_running("matrix_multiplication_1", as_gpu, bs_gpu, cs_gpu, gpu::WorkSize(16, 16, N, M),
                M, N, K, benchmarkingIters, gflops, cs);
    if (!check_result(M, N, cs, cs_cpu_reference)) {
        return 1;
    }

    gpu_running("matrix_multiplication_2", as_gpu, bs_gpu, cs_gpu, gpu::WorkSize(4, 16, N / 4, M),
                M, N, K, benchmarkingIters, gflops, cs);
    if (!check_result(M, N, cs, cs_cpu_reference)) {
        return 1;
    }

    return 0;
}
