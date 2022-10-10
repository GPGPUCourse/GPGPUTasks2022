#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 16
__kernel void matrix_multiplication(__global const float* a, __global const float* b, __global float* c,
                                    const unsigned int M, const unsigned int K, const unsigned int N){
   int i = get_global_id(0);
   int j = get_global_id(1);
   int local_i = get_local_id(0);
   int local_j = get_local_id(1);

   __local float local_a[TILE_SIZE][TILE_SIZE];
   __local float local_b[TILE_SIZE][TILE_SIZE];
   float sum = 0;
   for (size_t step = 0; step * TILE_SIZE < K; step++) {
      local_a[local_i][local_j] = a[i * K + local_j + step * TILE_SIZE];
      local_b[local_i][local_j] = b[(local_i + step * TILE_SIZE) * N + j];
      barrier(CLK_LOCAL_MEM_FENCE);

      // update local_c
      for (size_t index = 0; index < TILE_SIZE; index++) {
         sum += local_a[local_i][index] * local_b[index][local_j];
      }

      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (i < M && j < N) {
      c[i * N + j] = sum;
   }
}