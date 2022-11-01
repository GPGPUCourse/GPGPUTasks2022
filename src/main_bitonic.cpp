#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/bitonic_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

template <typename T>
T getDeviceInfo(cl_device_id deviceId, cl_device_info paramName) {
    T res;
    OCL_SAFE_CALL(clGetDeviceInfo(deviceId, paramName, sizeof(T), &res, nullptr));
    return res;
}

template <typename T>
std::vector<T> getArrayDeviceInfo(cl_device_id deviceId, cl_device_info paramName) {
    size_t valLength = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(deviceId, paramName, 0, nullptr, &valLength));
    std::vector<T> info(valLength);
    OCL_SAFE_CALL(clGetDeviceInfo(deviceId, paramName, valLength, info.data(), nullptr));
    return info;
}


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<float> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);
    int localMemorySize = getDeviceInfo<cl_ulong>(device.device_id_opencl, CL_DEVICE_LOCAL_MEM_SIZE);
    std::vector<size_t> maxWorkItemSizes = getArrayDeviceInfo<size_t>(device.device_id_opencl, CL_DEVICE_MAX_WORK_ITEM_SIZES);
    unsigned int n2 = 1;
    while (n2 < n) {
        n2 *= 2;
    }
    as.resize(n2);
    for (int i = n; i < n2; i++) {
        as[i] = CL_FLT_MAX;
    }

    int occupancy = 8;
    int maxChunk = localMemorySize / sizeof(float) / occupancy;
    maxChunk = std::min(int(maxWorkItemSizes[0]) * 2, maxChunk);

    maxChunk = std::min(1024, maxChunk); // hardcoding limit, because intel do not support dynamic local memory

    std::cout << "Max chunk size: " << maxChunk << std::endl;
    {
        ocl::Kernel bitonic(bitonic_kernel, bitonic_kernel_length, "bitonic");
        bitonic.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            unsigned int curr_chunk_size = 2;
            unsigned int global_work_size = n2 / 2;

            while (true) {
                unsigned int curr_sub_chunk_size = curr_chunk_size;

                while (curr_sub_chunk_size > 1) {
                    bool use_local_memory = curr_sub_chunk_size < maxChunk && curr_sub_chunk_size > 2;
                    unsigned int workGroupSize = 128;

                    //int local_mem_arr_size = 0;
                    //ocl::OpenCLKernelArg arg = ocl::OpenCLKernelArg();
                    if (use_local_memory) {
                        workGroupSize = std::max((unsigned int)(128), curr_sub_chunk_size / 2);
                        //local_mem_arr_size = workGroupSize * 2 * sizeof(float);
                    }
                    //arg.size = local_mem_arr_size;
                    //arg.is_null = false;
                    //arg.value = NULL;


                    bitonic.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, n2,
                                 curr_sub_chunk_size, curr_chunk_size, int(use_local_memory));
                    if (use_local_memory) {
                        break;
                    }
                    curr_sub_chunk_size /= 2;
                }
                if (curr_chunk_size >= n2) {
                    break;
                }
                curr_chunk_size *= 2;
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
