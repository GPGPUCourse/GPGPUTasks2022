#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float* gpu_results,
                         unsigned int width,
                         unsigned int height,
                         float fromX, float fromY,
                         float sizeX, float sizeY,
                         unsigned int iters,
                         int smoothing)
{
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    const unsigned int idx = get_global_id(0);
    const unsigned int idy = get_global_id(1);

    float x0 = fromX + ((float) idx + 0.5f) * sizeX / width;
    float y0 = fromY + ((float) idy + 0.5f) * sizeY / height;

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

    float result = iter;
    result = 1.0f * result / iters;
    gpu_results[idy * width + idx] = result;
}
