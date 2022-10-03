#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float* gpu, const unsigned int width, const unsigned int height,
                         const float fromX, const float fromY, const float sizeX, const float sizeY,
                         const unsigned int iterationsLimit, const int smoothing) {
   const float threshold = 256.0f * 256.0f;

   const unsigned int i = get_global_id(0);
   const unsigned int j = get_global_id(1);
   float x0 = fromX + (i + 0.5f) * sizeX / width;
   float y0 = fromY + (j + 0.5f) * sizeY / height;

   float x = x0;
   float y = y0;

   int iter = 0;
   for (; iter < iterationsLimit; ++iter) {
      float xPrev = x;
      x = x * x - y * y + x0;
      y = 2.0f * xPrev * y + y0;
      if ((x * x + y * y) > threshold) {
         break;
      }
   }
   float result = iter;
   if (smoothing && iter != iterationsLimit) {
      result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
   }

   result = 1.0f * result / iterationsLimit;
   gpu[j * width + i] = result;
}
