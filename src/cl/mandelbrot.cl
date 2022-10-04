#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *results,
                         unsigned int width,
                         unsigned int height,
                         float fromX,
                         float fromY,
                         float sizeX,
                         float sizeY,
                         unsigned int maxIter)
{

    const float THRESHOLD = 256.0f;
    const float THRESHOLD2 = THRESHOLD * THRESHOLD;

    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    float x0 = fromX + (i + 0.5f) * sizeX / width;
    float y0 = fromY + (j + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    int iter = 0;
    for (; iter < maxIter; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;

        if ((x * x + y * y) > THRESHOLD2) {
          break;
        }
    }

    float result = iter;

    if (iter != maxIter) {
        result -= log(log(sqrt(x * x + y * y)) / log(THRESHOLD)) / log(2.0f);
    }

    results[j * width + i] = result / maxIter;
    // если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
}
