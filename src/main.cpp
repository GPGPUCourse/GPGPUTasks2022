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
#include <unordered_map>

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

enum DeviceType {
    CPU,
    GPU,
    Accelerator
};

struct Device {
    cl_platform_id platformId;
    cl_device_id deviceId;
};

class DeviceSelector {
public:
    DeviceSelector() {
        std::vector<cl_platform_id> platforms = getPlatforms();

        for (cl_platform_id platform : platforms) {
            std::vector<cl_device_id> devices = getDevices(platform);
            distributeDevices(platform, devices);
        }
    }

    ~DeviceSelector() = default;


    Device pick(DeviceType deviceType) {
        if (_devices[deviceType].empty()) throw std::runtime_error("No such device type");

        return _devices[deviceType].front();
    }

    Device tryPickGPU() {
        try {
            return pick(DeviceType::GPU);
        } catch (std::runtime_error &e) {
            if (_devices[DeviceType::CPU].empty()) throw std::runtime_error("No CPU available");

            return _devices[DeviceType::CPU].front();
        }
    }

private:
    std::vector<cl_platform_id> getPlatforms() {
        cl_uint platformsCount = 0;
        OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));

        std::vector<cl_platform_id> platforms(platformsCount);
        OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

        return platforms;
    }

    std::vector<cl_device_id> getDevices(cl_platform_id platform) {
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        return devices;
    }

    std::vector<DeviceType> getDeviceType(cl_device_id device) {
        cl_device_type deviceTypes = 0;
        OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceTypes, nullptr));
        std::vector<DeviceType> types = parseDeviceTypes(deviceTypes);
        return types;
    }

    void distributeDevices(cl_platform_id platform, std::vector<cl_device_id> devices) {
        for (cl_device_id device : devices) {
            std::vector<DeviceType> deviceTypes = getDeviceType(device);

            for (DeviceType type : deviceTypes) {
                _devices[type].push_back({platform, device});
            }
        }
    }

    std::vector<DeviceType> parseDeviceTypes(cl_device_type deviceTypes) {
        std::vector<DeviceType> deviceTypeStrings;
        if ((deviceTypes & CL_DEVICE_TYPE_CPU) != 0) {
            deviceTypeStrings.emplace_back(DeviceType::CPU);
        }

        if ((deviceTypes & CL_DEVICE_TYPE_GPU) != 0) {
            deviceTypeStrings.emplace_back(DeviceType::GPU);
        }

        if ((deviceTypes & CL_DEVICE_TYPE_ACCELERATOR) != 0) {
            deviceTypeStrings.emplace_back(DeviceType::Accelerator);
        }

        return deviceTypeStrings;
    }

    std::unordered_map<DeviceType, std::vector<Device>> _devices;
};

void printDeviceName(cl_device_id deviceId) {
    size_t deviceNameSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));

    std::vector<unsigned char> deviceName(deviceNameSize, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
    std::cout << "Device name: " << deviceName.data() << std::endl;
}

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    cl_int err;
    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    DeviceSelector selector;
    Device dev = selector.tryPickGPU();
    printDeviceName(dev.deviceId);


    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    cl_context context = clCreateContext(nullptr, 1, &dev.deviceId, nullptr, nullptr, &err);
    OCL_SAFE_CALL(err);

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)
    cl_command_queue queue = clCreateCommandQueue(context, dev.deviceId, 0, &err);
    OCL_SAFE_CALL(err);


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

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично, все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)
    cl_mem as_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, as.size() * sizeof(float), as.data(), &err);
    OCL_SAFE_CALL(err);

    cl_mem bs_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bs.size() * sizeof(float), bs.data(), &err);
    OCL_SAFE_CALL(err);

    cl_mem cs_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY, cs.size() * sizeof(float), nullptr, &err);
    OCL_SAFE_CALL(err);


    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        // std::cout << kernel_sources << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель
    const char *sources = kernel_sources.c_str();
    cl_uint cnt = 1;
    size_t lengths = kernel_sources.length();
    cl_program program = clCreateProgramWithSource(context, cnt, &sources, &lengths, &err);
    OCL_SAFE_CALL(err);

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    try {
        OCL_SAFE_CALL(clBuildProgram(program, 1, &dev.deviceId, nullptr, nullptr, nullptr));
    } catch (std::runtime_error &e) {
        size_t logSize = 0;
        OCL_SAFE_CALL(clGetProgramBuildInfo(program, dev.deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize));

        std::vector<char> log(logSize, 0);
        OCL_SAFE_CALL(clGetProgramBuildInfo(program, dev.deviceId, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr));

        if (logSize > 1) {
            std::cerr << "Log:" << std::endl;
            std::cerr << log.data() << std::endl;
        }

        throw e;
    }

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    cl_kernel kernel = clCreateKernel(program, "aplusb", &err);
    OCL_SAFE_CALL(err);


    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
         unsigned int i = 0;
         clSetKernelArg(kernel, i++, sizeof(cl_mem), &as_gpu);
         clSetKernelArg(kernel, i++, sizeof(cl_mem), &bs_gpu);
         clSetKernelArg(kernel, i++, sizeof(cl_mem), &cs_gpu);
         clSetKernelArg(kernel, i++, sizeof(unsigned int), &n);
    }

    // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)
    //Done

    // TODO 12 Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число, кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание, что, чтобы дождаться окончания вычислений (чтобы знать, когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;// Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, &workGroupSize, 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // TODO 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << ((float) n / t.lapAvg()) / 1e9 << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << ((float) (3 * n * sizeof(float)) / t.lapAvg()) / 1024 / 1024 / 1024 << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(queue, cs_gpu, CL_TRUE, 0, n * sizeof(float), cs.data(), 0, nullptr, nullptr));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << (float) (n * sizeof(float)) / t.lapAvg() / 1024 / 1024 / 1024 << " GB/s" << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ: " + std::to_string(cs[i]) + " != " + std::to_string(as[i]) + " + " + std::to_string(bs[i]));
        }
    }

    OCL_SAFE_CALL(clReleaseKernel(kernel));
    OCL_SAFE_CALL(clReleaseProgram(program));
    OCL_SAFE_CALL(clReleaseMemObject(as_gpu));
    OCL_SAFE_CALL(clReleaseMemObject(bs_gpu));
    OCL_SAFE_CALL(clReleaseMemObject(cs_gpu));
    OCL_SAFE_CALL(clReleaseCommandQueue(queue));
    OCL_SAFE_CALL(clReleaseContext(context));

    return 0;
}

