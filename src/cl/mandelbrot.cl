#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6


float calcPoint(float x0, float y0, uint iters) {
  const float threshold2 = 4.0f;

  float x = x0;
  float y = y0;

  uint iter = 0;
  for (; iter < iters; ++iter) {
    float xPrev = x;
    x = x * x - y * y + x0;
    y = 2.0f * xPrev * y + y0;
    if ((x * x + y * y) > threshold2) {
      break;
    }
  }

  return 1.0f * iter / iters;
}

float calcPointAA(float x0, float y0, float scale, uint iters, uint aaIters) {
  if (aaIters < 2) {
    return calcPoint(x0, y0, iters);
  }
  float r = scale / 4;
  float angle = M_PI_F / aaIters;
  float res = 0.0;
  for (uint i = 0; i < aaIters; ++i) {
    res += calcPoint(x0 + r * sin(i * angle), y0 + r * cos(i * angle), iters);
  }
  return res / aaIters;
}

__kernel void mandelbrot(
    __global float* image,
    float startX, float startY, uint width, uint height, float scale,
    uint iters, uint aaIters
) {
  size_t pixelX = get_global_id(0);
  size_t pixelY = get_global_id(1);
  if (pixelX >= width || pixelY >= height) {
    return;
  }

  float x = startX + scale * (pixelX + 0.5);
  float y = startY + scale * (pixelY + 0.5);

  image[pixelY * width + pixelX] = calcPointAA(x, y, scale, iters, aaIters);
}
