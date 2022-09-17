#include <CL/cl.h>
#include <libclew/ocl_init.h>

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
    // std::cout << message;  // Добавил, чтобы видеть код ошибки в первом задании
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // Откройте
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
    // Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    // Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
        // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
        size_t platformNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
        // TODO 1.1
        // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
        // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
        // Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
        // platformNameSize = 0;
        // OCL_SAFE_CALL(clGetPlatformInfo(platform, 239, 0, nullptr, &platformNameSize));
        // Откройте таблицу с кодами ошибок:
        // libs/clew/CL/cl.h:103
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        // Найдите там нужный код ошибки и ее название
        // Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
        // в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
        // Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.
        // Код ошибки: -30, её название: CL_INVALID_VALUE
        // Из секции Errors документации по clGetPlatformInfo видим, что ошибка вызвана некорректным аргументом param_name

        // TODO 1.2
        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::vector<unsigned char> platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        size_t platformVendorSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
        std::vector<unsigned char> platformVendor(platformVendorSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
        std::cout << "    Vendor: " << platformVendor.data() << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::cout << "    Number of available devices of this platform: " << devicesCount << std::endl;
        // Также для следующего задания создадим вектор из ID доступных устройств данной платформы
        std::vector<cl_device_id> platformDevices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, platformDevices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            // TODO 2.2
            std::cout << "        Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
            cl_device_id device = platformDevices[deviceIndex];
            // Запросите и напечатайте в консоль:
            // - Название устройства
            size_t deviceNameSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
            std::vector<unsigned char> deviceName(deviceNameSize, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
            std::cout << "            Device name: " << deviceName.data() << std::endl;
            // - Тип устройства (видеокарта/процессор/что-то странное)
            size_t deviceTypeSize = (1 << 3);  // Число взято из документации
            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, deviceTypeSize, &deviceType, nullptr));
            std::cout << "            Device type: ";
            if (deviceType == CL_DEVICE_TYPE_CPU) {
                std::cout << "CPU" << std::endl;
            }
            else if (deviceType == CL_DEVICE_TYPE_GPU) {
                std::cout << "GPU" << std::endl;
            }
            else {
                std::cout << "something else" << std::endl;
            }
            // - Размер памяти устройства в мегабайтах
            cl_ulong deviceMemory = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, 8, &deviceMemory, nullptr));
            std::cout << "            Memory size: " << deviceMemory / (1 << 20) << " MB" << std::endl;
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            // Максимальная тактовая частота устройства
            cl_uint deviceMaxClockFreq = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, 4, &deviceMaxClockFreq, nullptr));
            std::cout << "            Max clock frequency: " << deviceMaxClockFreq << " MHz" << std::endl;
            // Размер максимальной рабочей группы
            size_t deviceMaxWorkGroupSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 8, &deviceMaxWorkGroupSize, nullptr));
            std::cout << "            Max work group size: " << deviceMaxWorkGroupSize << std::endl;

        }
    }

    return 0;
}
