#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GROUP_SIDE_SIZE 32
#define GROUP_TILE_SIZE 8

__kernel void matrix_multiplication_fma(__global const float *as, __global const float *bs, __global float *cs, unsigned int M, unsigned int K, unsigned int N) {
    __local float storage_a[GROUP_SIDE_SIZE][GROUP_SIDE_SIZE];
    __local float storage_b[GROUP_SIDE_SIZE][GROUP_SIDE_SIZE];

    unsigned int local_x = get_local_id(0);
    unsigned int local_y = get_local_id(1);
    unsigned int global_x = get_global_id(0);
    unsigned int global_y = get_global_id(1);

    float res[GROUP_TILE_SIZE];
    for (unsigned int i = 0; i < GROUP_TILE_SIZE; i++) {
        res[i] = 0.0;
    }
    for (unsigned int i = 0; i < K; i += GROUP_SIDE_SIZE) {
        for (unsigned int j = 0; j < GROUP_TILE_SIZE; j++) {
            unsigned int xa = i + local_x;
            unsigned int ya = global_y * GROUP_TILE_SIZE + j;
            storage_a[local_y * GROUP_TILE_SIZE + j][local_x] = as[ya * K + xa];
        }

        for (unsigned  int j = 0; j < GROUP_TILE_SIZE; j++) {
            unsigned int xb = global_x;
            unsigned int yb = i + local_y * GROUP_TILE_SIZE + j;
            storage_b[local_y * GROUP_TILE_SIZE + j][local_x] = bs[yb * N + xb];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int j = 0; j < GROUP_SIDE_SIZE; j++) {
            float c = storage_b[j][local_x];
            for (unsigned int k = 0; k < GROUP_TILE_SIZE; k++) {
                res[k] += c * storage_a[local_y * GROUP_TILE_SIZE + k][j];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (unsigned int j = 0; j < GROUP_TILE_SIZE; j++) {
        cs[(global_y * GROUP_TILE_SIZE + j) * N + global_x] = res[j];
    }
}