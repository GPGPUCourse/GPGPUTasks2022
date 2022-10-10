#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 16
__kernel void matrix_transpose(__global const float* src, __global float* dest, unsigned int n, unsigned int m) {
   int i = get_global_id(0);
   int j = get_global_id(1);

   __local float tile[TILE_SIZE][TILE_SIZE + 1];
   int local_i = get_local_id(0);
   int local_j = get_local_id(1);

   if (i < n && j < m) {
      tile[local_j][local_i] = src[i * n + j];
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   if (i < n && j < m) {
      dest[j * m + i] = tile[local_j][local_i];
   }
}