#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 16
__kernel void matrix_transpose(__global const float* as,
                               __global float* as_t,
                               unsigned int M,
                               unsigned int K)
{
    int global_index_1 = get_global_id(0);
    int global_index_2 = get_global_id(1);
    int local_index_1 = get_local_id(0);
    int local_index_2 = get_local_id(1);

    if (global_index_1 >= K || global_index_2 >= M)
        return;

    __local float las[256];
    las[local_index_1 * TILE_SIZE + local_index_2] = as[global_index_1 * M + global_index_2];

    barrier(CLK_LOCAL_MEM_FENCE);
    __local float las_t[256];
    las_t[local_index_2 * TILE_SIZE + local_index_1] = las[local_index_1 * TILE_SIZE + local_index_2];
    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[global_index_2 * K + global_index_1] = las_t[local_index_2 * TILE_SIZE + local_index_1];

}