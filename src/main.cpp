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

void errReleaseContext(cl_int error) {
    if (error != CL_SUCCESS) {
        std::cout << "Error release context with exit code: " << error << std::endl;
    }
}

void errReleaseQueue(cl_int error) {
    if (error != CL_SUCCESS) {
        std::cout << "Error release queue with exit code: " << error << std::endl;
    }
}

void errReleaseBuffer(cl_int error) {
    if (error != CL_SUCCESS) {
        std::cout << "Error release buffer with exit code: " << error << std::endl;
    }
}

void errReleaseProgram(cl_int error) {
    if (error != CL_SUCCESS) {
        std::cout << "Error release program with exit code: " << error << std::endl;
    }
}

void errReleaseKernel(cl_int error) {
    if (error != CL_SUCCESS) {
        std::cout << "Error release kernel with exit code: " << error << std::endl;
    }
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
    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)

    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));

    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    cl_device_id device;
    cl_device_type deviceType = CL_DEVICE_TYPE_CPU;

    for (int platformInd = 0; platformInd < platforms.size() && deviceType != CL_DEVICE_TYPE_GPU; ++platformInd) {
        cl_uint deviceCount = 0;
        std::vector<cl_device_id> devices;

        OCL_SAFE_CALL(clGetDeviceIDs(platforms[platformInd], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount));
        devices.resize(deviceCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platforms[platformInd], CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr));

        for (int deviceInd = 0; deviceInd < devices.size() && deviceType != CL_DEVICE_TYPE_GPU; ++deviceInd) {
            OCL_SAFE_CALL(
                    clGetDeviceInfo(devices[deviceInd], CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
            device = devices[deviceInd];

        }
    }

    size_t name = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &name));
    std::vector<unsigned char> deviceName(name, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, name, deviceName.data(), nullptr));
    std::cout << "device name: " << deviceName.data() << std::endl;

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)

    cl_int errorCreateContext = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &errorCreateContext);

    if (errorCreateContext != CL_SUCCESS) {
        cl_int errorReleaseContext = clReleaseContext(context);
        errReleaseContext(errorReleaseContext);
        switch (errorCreateContext) {
            case CL_INVALID_VALUE:
                std::cout << "CL_INVALID_VALUE" << std::endl;
                break;
            case CL_DEVICE_NOT_AVAILABLE:
                std::cout << "CL_DEVICE_NOT_AVAILABLE" << std::endl;
                break;
            case CL_INVALID_OPERATION:
                std::cout << "CL_INVALID_OPERATION" << std::endl;
                break;
            default:
                std::cout << "Error create context with exit code: " << errorCreateContext << std::endl;
                break;
        }
    }

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)
    cl_int errorCreateQueue = CL_SUCCESS;
    cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &errorCreateQueue);
    if (errorCreateQueue != CL_SUCCESS) {
        cl_int errorReleaseQueue = clReleaseCommandQueue(commandQueue);
        errReleaseQueue(errorReleaseQueue);

        cl_int errorReleaseContext = clReleaseContext(context);
        errReleaseContext(errorReleaseContext);

        std::cout << "Error create queue with exit code: " << errorCreateQueue << std::endl;
    }

    cl_command_queue_properties commandQueueProperties;
    OCL_SAFE_CALL(clGetCommandQueueInfo(
                commandQueue, CL_QUEUE_PROPERTIES,
                sizeof(cl_command_queue_properties),
                &commandQueueProperties, nullptr
            ));


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

    cl_int errorCreateBuffer = CL_SUCCESS;
    cl_mem memA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), bs.data(),
                                 &errorCreateBuffer);
    if (errorCreateBuffer != CL_SUCCESS) {
        cl_int errorReleaseBuffer = clReleaseMemObject(memA);
        errReleaseBuffer(errorReleaseBuffer);

        cl_int errorReleaseQueue = clReleaseCommandQueue(commandQueue);
        errReleaseQueue(errorReleaseQueue);

        cl_int errorReleaseContext = clReleaseContext(context);
        errReleaseContext(errorReleaseContext);

        std::cout << "Error create buffer A with exit code: " << errorCreateBuffer << std::endl;
    }

    cl_mem memB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), bs.data(),
                                 &errorCreateBuffer);
    if (errorCreateBuffer != CL_SUCCESS) {
        cl_int errorReleaseBuffer = clReleaseMemObject(memB);
        errReleaseBuffer(errorReleaseBuffer);

        errorReleaseBuffer = clReleaseMemObject(memA);
        errReleaseBuffer(errorReleaseBuffer);

        cl_int errorReleaseQueue = clReleaseCommandQueue(commandQueue);
        errReleaseQueue(errorReleaseQueue);

        cl_int errorReleaseContext = clReleaseContext(context);
        errReleaseContext(errorReleaseContext);

        std::cout << "Error create buffer B with exit code: " << errorCreateBuffer << std::endl;
    }

    cl_mem memC = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), bs.data(),
                                 &errorCreateBuffer);
    if (errorCreateBuffer != CL_SUCCESS) {
        cl_int errorReleaseBuffer = clReleaseMemObject(memC);
        errReleaseBuffer(errorReleaseBuffer);

        errorReleaseBuffer = clReleaseMemObject(memB);
        errReleaseBuffer(errorReleaseBuffer);

        errorReleaseBuffer = clReleaseMemObject(memA);
        errReleaseBuffer(errorReleaseBuffer);

        cl_int errorReleaseQueue = clReleaseCommandQueue(commandQueue);
        errReleaseQueue(errorReleaseQueue);

        cl_int errorReleaseContext = clReleaseContext(context);
        errReleaseContext(errorReleaseContext);

        std::cout << "Error create buffer C with exit code: " << errorCreateBuffer << std::endl;
    }

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

    cl_int error = CL_SUCCESS;
    const char *sourceCode = kernel_sources.data();
    const size_t codeLen = kernel_sources.length();
    cl_program program = clCreateProgramWithSource(context, 1, &sourceCode, &codeLen, &error);

    if (error != CL_SUCCESS) {
        cl_int errorReleaseProgram = clReleaseProgram(program);
        errReleaseProgram(errorReleaseProgram);

        cl_int errorReleaseBuffer = clReleaseMemObject(memC);
        errReleaseBuffer(errorReleaseBuffer);

        errorReleaseBuffer = clReleaseMemObject(memB);
        errReleaseBuffer(errorReleaseBuffer);

        errorReleaseBuffer = clReleaseMemObject(memA);
        errReleaseBuffer(errorReleaseBuffer);

        cl_int errorReleaseQueue = clReleaseCommandQueue(commandQueue);
        errReleaseQueue(errorReleaseQueue);

        cl_int errorReleaseContext = clReleaseContext(context);
        errReleaseContext(errorReleaseContext);

        std::cout << "Error create program with exit code: " << error << std::endl;
    }

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram

    cl_int build = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (build != CL_SUCCESS) {
        cl_int errorReleaseProgram = clReleaseProgram(program);
        errReleaseProgram(errorReleaseProgram);

        cl_int errorReleaseBuffer = clReleaseMemObject(memC);
        errReleaseBuffer(errorReleaseBuffer);

        errorReleaseBuffer = clReleaseMemObject(memB);
        errReleaseBuffer(errorReleaseBuffer);

        errorReleaseBuffer = clReleaseMemObject(memA);
        errReleaseBuffer(errorReleaseBuffer);

        cl_int errorReleaseQueue = clReleaseCommandQueue(commandQueue);
        errReleaseQueue(errorReleaseQueue);

        cl_int errorReleaseContext = clReleaseContext(context);
        errReleaseContext(errorReleaseContext);

        std::cout << "Error build program with exit code: " << build << std::endl;
    }

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo

    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
    std::vector<char> log(log_size + 1, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr));
    if (log_size > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    }

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects

    cl_kernel kernel = clCreateKernel(program, "aplusb", &error);
    if (error != CL_SUCCESS) {
        cl_int errorReleaseKernel = clReleaseKernel(kernel);
        errReleaseKernel(errorReleaseKernel);

        cl_int errorReleaseProgram = clReleaseProgram(program);
        errReleaseProgram(errorReleaseProgram);

        cl_int errorReleaseBuffer = clReleaseMemObject(memC);
        errReleaseBuffer(errorReleaseBuffer);

        errorReleaseBuffer = clReleaseMemObject(memB);
        errReleaseBuffer(errorReleaseBuffer);

        errorReleaseBuffer = clReleaseMemObject(memA);
        errReleaseBuffer(errorReleaseBuffer);

        cl_int errorReleaseQueue = clReleaseCommandQueue(commandQueue);
        errReleaseQueue(errorReleaseQueue);

        cl_int errorReleaseContext = clReleaseContext(context);
        errReleaseContext(errorReleaseContext);

        std::cout << "Error create kernel with exit code: " << error << std::endl;
    }
    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &memA));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &memB));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &memC));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(n), &n));
    }

    // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

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
        cl_event event;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(
                        commandQueue, kernel, 1,
                        nullptr, &global_work_size,
                        &workGroupSize, 0, nullptr,&event
                    ));
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
        std::cout << "GFlops: " << n / t.lapAvg() / 10e8 << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 3.0 * n + sizeof(float) / (1024 * 1024 * 1024) / t.lapAvg() << " GB/s"
                  << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(
                        commandQueue, memC,
                        CL_TRUE, 0, n * sizeof(float),
                        cs.data(), 0, nullptr, nullptr
                    ));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << n * sizeof(float) / t.lapAvg() / (1024 * 1024 * 1024) << " GB/s"
                  << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            std::cout << as[i] + bs[i] << std::endl;
            std::cout << cs[i] << std::endl;
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    return 0;
}
