#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 128
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

__kernel void matrix_transpose(
    __global const float *matrix,
    __global float *result,
    unsigned int M,
    unsigned int K
) {
    const int local_id = get_local_id(0);
    const int group_id = get_group_id(0);

    const int groups_by_x = (K + TILE_SIZE - 1) / TILE_SIZE;
    const int group_x = group_id % groups_by_x;
    const int group_y = group_id / groups_by_x;
    const int offset_x = group_x * TILE_SIZE;
    const int offset_y = group_y * TILE_SIZE;

    __local float uploaded[TILE_SIZE * (TILE_SIZE + 1)];

    for (int i = local_id; i < TILE_SIZE * TILE_SIZE; i += WORK_GROUP_SIZE) {
        int x = i % TILE_SIZE;
        int y = i / TILE_SIZE;
        int to = x + y * (TILE_SIZE + 1);

        int global_x = offset_x + x;
        int global_y = offset_y + y;
        int from = global_x + global_y * K;

        if (global_x < K && global_y < M) {
            uploaded[to] = matrix[from];
        } else {
            uploaded[to] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = local_id; i < TILE_SIZE * TILE_SIZE; i += WORK_GROUP_SIZE) {
        int y = i % TILE_SIZE;
        int x = i / TILE_SIZE;
        int from = x + y * (TILE_SIZE + 1);

        int global_x = offset_x + x;
        int global_y = offset_y + y;
        int to = global_x * M + global_y;

        if (global_x < K && global_y < M) {
            result[to] = uploaded[from];
        }
    }
}