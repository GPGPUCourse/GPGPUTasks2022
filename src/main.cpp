#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    bool gpu_is_found = false;
    cl_device_id device = nullptr;
    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        cl_platform_id platform = platforms[platformIndex];

        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            device = devices[deviceIndex];

            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
            if (deviceType == CL_DEVICE_TYPE_GPU) {
                gpu_is_found = true;
                break;
            }
        }
        if (gpu_is_found) {
            break;
        }
    }

    assert(device != nullptr);

    cl_int errcode_ret = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    unsigned int n = 100 * 1000 * 1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    cl_mem a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, as.data(),
                              &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    cl_mem b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, bs.data(),
                              &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    cl_mem c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.empty()) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        // std::cout << kernel_sources << std::endl;
    }

    const char *pointer = kernel_sources.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &pointer, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    OCL_SAFE_CALL(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr));

    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));

    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), &log_size));
    assert(log_size == log.size());

    if (log_size > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    }

    cl_kernel kernel = clCreateKernel(program, "aplusb", &errcode_ret);

    {
        unsigned int i = 0;
        clSetKernelArg(kernel, i++, sizeof(cl_mem), &a);
        clSetKernelArg(kernel, i++, sizeof(cl_mem), &b);
        clSetKernelArg(kernel, i++, sizeof(cl_mem), &c);
        clSetKernelArg(kernel, i, sizeof(unsigned int), &n);
    }

    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;// Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            // clEnqueueNDRangeKernel...
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, &workGroupSize, 0,
                                                 nullptr, &event));
            // clWaitForEvents...
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        std::cout << "GFlops: " << n / t.lapAvg() / 1e9 << std::endl;

        std::cout << "VRAM bandwidth: " << (double) 3 * sizeof(float) * n / 1024 / 1024 / 1024 / t.lapAvg() << " GB/s"
                  << std::endl;
    }

    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(queue, c, CL_TRUE, 0, sizeof(float) * n, cs.data(), 0, nullptr, nullptr));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << (double) sizeof(float) * n / 1024 / 1024 / 1024 / t.lapAvg()
                  << " GB/s" << std::endl;
    }

    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    OCL_SAFE_CALL(clReleaseKernel(kernel));
    OCL_SAFE_CALL(clReleaseProgram(program));
    OCL_SAFE_CALL(clReleaseMemObject(c));
    OCL_SAFE_CALL(clReleaseMemObject(b));
    OCL_SAFE_CALL(clReleaseMemObject(a));
    OCL_SAFE_CALL(clReleaseCommandQueue(queue));
    OCL_SAFE_CALL(clReleaseContext(context));

    return 0;
}
