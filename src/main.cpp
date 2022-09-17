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
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

static std::string queryPlatformInfo(cl_platform_id platform, cl_platform_info param) {
    size_t paramSize = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, param, 0, nullptr, &paramSize));
    std::vector<char> data(paramSize);
    OCL_SAFE_CALL(clGetPlatformInfo(platform, param, paramSize, data.data(), nullptr));
    return data.data();
}

static std::string queryDeviceInfo(cl_device_id device, cl_device_info param) {
    size_t paramSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, param, 0, nullptr, &paramSize));
    std::vector<char> data(paramSize);
    OCL_SAFE_CALL(clGetDeviceInfo(device, param, paramSize, data.data(), nullptr));
    return data.data();
}

template <class T>
static T queryDeviceInfo(cl_device_id device, cl_device_info param) {
    T res;
    size_t size;
    OCL_SAFE_CALL(clGetDeviceInfo(device, param, sizeof(T), &res, &size));
    if (size != sizeof(T)) {
        throw std::runtime_error("Size of device param (" + to_string(param) + ") "
                                     "does not match the expected value");
    }
    return res;
}

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
        try {
            OCL_SAFE_CALL(clGetPlatformInfo(platform, 239, 0, nullptr, &platformNameSize));
        } catch (const std::runtime_error& e) {
            std::cerr << e.what() << std::endl;
        }
        // _TODO 1.1
        // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
        // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
        // Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
        // Откройте таблицу с кодами ошибок:
        // libs/clew/CL/cl.h:103
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        // Найдите там нужный код ошибки и ее название
        // Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
        // в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
        // Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

        // _TODO 1.2
        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::cout << "    Platform name: " << queryPlatformInfo(platform, CL_PLATFORM_NAME) << std::endl;

        // _TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        std::cout << "    Platform vendor: " << queryPlatformInfo(platform, CL_PLATFORM_VENDOR) << std::endl;

        // _TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::vector<cl_device_id> deviceIds(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, deviceIds.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
            cl_device_id device = deviceIds[deviceIndex];
            std::cout << "        Device name: " << queryDeviceInfo(device, CL_DEVICE_NAME) << std::endl;

            cl_device_type deviceType = queryDeviceInfo<cl_device_type>(device, CL_DEVICE_TYPE);
            std::string deviceTypeString = "";
            if (deviceType & CL_DEVICE_TYPE_CPU)
                deviceTypeString += "cpu/";
            if (deviceType & CL_DEVICE_TYPE_GPU)
                deviceTypeString += "gpu/";
            if (deviceType & CL_DEVICE_TYPE_ACCELERATOR)
                deviceTypeString += "accelerator/";
            if (deviceTypeString.empty())
                deviceTypeString = "custom";
            else
                deviceTypeString.resize(deviceTypeString.size() - 1);

            std::cout << "        Device type: " << deviceType << " -> " << deviceTypeString << std::endl;

            cl_ulong deviceMem = queryDeviceInfo<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
            std::cout << "        Device memory: " << deviceMem / (1024*1024) << " MiB" << std::endl;

            // _TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными

            std::cout << "        Device C version: " << queryDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION) << std::endl;
            std::cout << "        Device host unified memory: " << std::boolalpha << (queryDeviceInfo<cl_bool>(device, CL_DEVICE_HOST_UNIFIED_MEMORY) == CL_TRUE) << std::endl;
        }
    }

    return 0;
}
