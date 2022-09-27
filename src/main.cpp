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

cl_int get_device(cl_device_type deviceType, cl_platform_id *dst_platform, cl_device_id* dst_device) {
    cl_uint platformCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformCount));
    std::vector <cl_platform_id> platforms(platformCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformCount, platforms.data(), nullptr));

    for (size_t platformIndex = 0; platformIndex < platformCount; platformIndex++) {
        cl_platform_id platform = platforms[platformIndex];

        cl_uint deviceCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, deviceType, 0, nullptr, &deviceCount));

        if (deviceCount == 0)
            continue;

        *dst_platform = platform;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, deviceType, 1, dst_device, nullptr));
        return CL_SUCCESS;
    }

    return CL_DEVICE_NOT_FOUND;
}


int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        size_t platformNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
        std::vector<unsigned char> platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        size_t platformVendorSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
        std::vector<unsigned char> platformVendor(platformVendorSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformName.data(), nullptr));
        std::cout << "    Platform vendor: " << platformName.data() << std::endl;

        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount);
        std::cout << "    Platform device count: " << devicesCount << std::endl;
        std::vector<cl_device_id> devices(platformsCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;

            cl_device_id device = devices[deviceIndex];
            // Запросите и напечатайте в консоль:
            // - Название устройства
            size_t deviceNameSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
            std::vector<unsigned char> deviceName(deviceNameSize, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
            std::cout << "        Device name: " << deviceName.data() << std::endl;

            // - Тип устройства (видеокарта/процессор/что-то странное)
            cl_device_type deviceType = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
            std::string deviceTypeName;
            if (deviceType == CL_DEVICE_TYPE_DEFAULT)
                deviceTypeName = "DEFAULT";
            if (deviceType & CL_DEVICE_TYPE_CPU)
                deviceTypeName += "CPU ";
            if (deviceType & CL_DEVICE_TYPE_GPU)
                deviceTypeName += "GPU ";
            if (deviceType & CL_DEVICE_TYPE_ACCELERATOR)
                deviceTypeName += "ACCELERATOR ";
            std::cout << "        Device type: " << deviceTypeName << std::endl;

            // - Размер памяти устройства в мегабайтах
            ulong deviceGlobalMemorySize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ulong), &deviceGlobalMemorySize, nullptr));
            std::cout << "        Global memory size: " << deviceGlobalMemorySize / (1024 * 1024) << std::endl;

            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            size_t deviceVersionSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, nullptr, &deviceVersionSize));
            std::vector<unsigned char> deviceVersion(deviceNameSize, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VERSION, deviceVersionSize, deviceVersion.data(), nullptr));
            std::cout << "        Device version: " << deviceVersion.data() << std::endl;
        }
    }

    // По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    cl_platform_id platform;
    cl_device_id device;
    if (get_device(CL_DEVICE_TYPE_GPU, &platform, &device) == CL_DEVICE_NOT_FOUND)
        if (get_device(CL_DEVICE_TYPE_CPU, &platform, &device) == CL_DEVICE_NOT_FOUND)
            OCL_SAFE_CALL(get_device(CL_DEVICE_TYPE_ALL, &platform, &device));

    // - Название устройства
    size_t deviceNameSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
    std::vector<unsigned char> deviceName(deviceNameSize, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
    std::cout << "Device name: " << deviceName.data() << std::endl;

    // - Тип устройства (видеокарта/процессор/что-то странное)
    cl_device_type deviceType = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
    std::string deviceTypeName;
    if (deviceType == CL_DEVICE_TYPE_DEFAULT)
        deviceTypeName = "DEFAULT";
    if (deviceType & CL_DEVICE_TYPE_CPU)
        deviceTypeName += "CPU ";
    if (deviceType & CL_DEVICE_TYPE_GPU)
        deviceTypeName += "GPU ";
    if (deviceType & CL_DEVICE_TYPE_ACCELERATOR)
        deviceTypeName += "ACCELERATOR ";
    std::cout << "Device type: " << deviceTypeName << std::endl;

    // Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    cl_int errcode_ret;
    cl_context_properties properties[] =
        {
                CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform),
                0 // signals end of property list
        };
    cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);


    // Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)

    cl_command_queue commandQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
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

    // Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично, все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)
    cl_mem as_gpu = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, as.size() * sizeof(float), as.data(), &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    cl_mem bs_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY, bs.size() * sizeof(float), nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    OCL_SAFE_CALL(clEnqueueWriteBuffer(commandQueue, bs_gpu, CL_TRUE, 0, bs.size() * sizeof(float), bs.data(), 0, nullptr, nullptr));
    cl_mem cs_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY , cs.size() * sizeof(float), nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    // Выполните (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        //std::cout << kernel_sources << std::endl;
    }

    // Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель
    const char *strings = kernel_sources.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &strings, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    // Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    OCL_SAFE_CALL(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr));

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    //    size_t log_size = 0;
    //    std::vector<char> log(log_size, 0);
    //    if (log_size > 1) {
    //        std::cout << "Log:" << std::endl;
    //        std::cout << log.data() << std::endl;
    //    }

    // Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    cl_kernel kernel = clCreateKernel(program, "aplusb", &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    // Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
         unsigned int i = 0;
         clSetKernelArg(kernel, i++, sizeof(as_gpu), &as_gpu);
         clSetKernelArg(kernel, i++, sizeof(bs_gpu), &bs_gpu);
         clSetKernelArg(kernel, i++, sizeof(cs_gpu), &cs_gpu);
         clSetKernelArg(kernel, i++, sizeof(n), &n);
    }

    // Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    // Запустите выполнения кернела:
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
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << n / t.lapAvg() / 1e9 << std::endl;

        // Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 3 * float(n) * sizeof(float) / t.lapAvg() / float(1024 * 1024 * 1024) << " GB/s" << std::endl;
    }

    // Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueReadBuffer(commandQueue, cs_gpu, CL_FALSE, 0, sizeof(float)*cs.size(), cs.data(),
                                              0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << n * sizeof(float) / t.lapAvg() / (1024*1024*1024) << " GB/s" << std::endl;
    }

    // Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    OCL_SAFE_CALL(clReleaseMemObject(as_gpu));
    OCL_SAFE_CALL(clReleaseMemObject(bs_gpu));
    OCL_SAFE_CALL(clReleaseMemObject(cs_gpu));
    OCL_SAFE_CALL(clReleaseCommandQueue(commandQueue));
    OCL_SAFE_CALL(clReleaseContext(context));

    return 0;
}
