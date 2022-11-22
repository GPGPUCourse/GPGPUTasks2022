#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float* results,
                         unsigned int width, unsigned int height,
                         float fromX, float fromY,
                         float sizeX, float sizeY,
                         unsigned int iters, unsigned int smoothing)
{
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен

    const unsigned int j = get_group_id(0);
    const unsigned int i = get_group_id(1);

    if (j >= height || i >= width)
        return;

    const unsigned int local_j = get_local_id(0);
    const unsigned int local_i = get_local_id(1);

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    __local int neighbors[16][16];

    float x0 = fromX + (i + (local_i) / get_local_size(1) + 0.5f) * sizeX / width;
    float y0 = fromY + (j + (local_j) / get_local_size(0) + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    int iter = 0;
    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2) {
            break;
        }
    }

    if (smoothing && iter != iters) {
        iter = iter - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }

    neighbors[local_j][local_i] = iter;

    if (local_i + local_j == 0) {
        float result = 0;
        for (int l_j = 0; l_j < get_local_size(0); l_j++) {
            for (int l_i = 0; l_i < get_local_size(1); l_i++) {
                result += neighbors[l_j][l_i];
            }
        }
        result /= get_local_size(0) * get_local_size(1);

        result = 1.0f * result / iters;
        results[j * width + i] = result;
    }
}
