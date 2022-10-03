#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__constant const float threshold = 256.0f;
__constant const float threshold2 = threshold * threshold;

__kernel void mandelbrot(__global float* results,
                         unsigned int width, unsigned int height,
                         float fromX, float fromY,
                         float sizeX, float sizeY,
                         unsigned int iters)
{
    int global_id = get_global_id(0);
    if (global_id >= width * height) {
        return;
    }

    unsigned int ys = global_id / width;
    unsigned int xs = global_id - ys * width;

    float x0 = fromX + (xs + 0.5f) * sizeX / width;
    float y0 = fromY + (ys + 0.5f) * sizeY / height;

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
    results[ys * width + xs] = result;
}
