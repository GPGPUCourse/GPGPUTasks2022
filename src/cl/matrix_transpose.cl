#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 16

__kernel void matrix_transpose(__global float *a, __global float *at, int m, int k) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float local_a[TILE_SIZE][TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    local_a[local_j][local_i] = a[j * k + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    int new_i = j - local_j + local_i;
    int new_j = i - local_i + local_j;
    if (new_i < m && new_j < k)
        at[new_j * m + new_i] = local_a[local_i][local_j];
}

