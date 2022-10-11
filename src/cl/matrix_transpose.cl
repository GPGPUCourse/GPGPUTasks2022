#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 16
__kernel void matrix_transpose(__global float *a, __global float *at, unsigned int m, unsigned int k) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if (i < m && j < k)
        tile[local_j][local_i] = a[i * m + j];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < m && j < k)
        at[j * k + i] = tile[local_j][local_i];
}