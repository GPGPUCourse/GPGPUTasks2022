#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 16

__kernel void matrix_multiplication_1(__global float *a, __global float *b, __global float *c, int M, int K, int N){
    __local float local_a[TILE_SIZE][TILE_SIZE];
    __local float local_b[TILE_SIZE][TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    int i = get_global_id(0);
    int j = get_global_id(1);
    float sum = 0.0f;
    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        local_a[local_j][local_i] = a[j * K + tileK * TILE_SIZE + local_i];
        local_b[local_j][local_i] = b[(tileK * TILE_SIZE + local_j) * N + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += local_a[local_j][k] * local_b[k][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[j * N + i] = sum;
}

#define TILE_SIZE_2 32
#define WPT 8
#define RTS TILE_SIZE_2/WPT

__kernel void matrix_multiplication_2(__global float *a, __global float *b, __global float *c, int M, int K, int N){
    __local float local_a[TILE_SIZE_2][TILE_SIZE_2];
    __local float local_b[TILE_SIZE_2][TILE_SIZE_2];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    int i = get_group_id(0) * TILE_SIZE_2 + local_i;
    int j = get_group_id(1) * TILE_SIZE_2 + local_j;
    float sum[WPT];
    for (int w = 0; w < WPT; w++) {
        sum[w] = 0.0f;
    }
    for (int tileK = 0; tileK * TILE_SIZE_2 < K; tileK++) {
        for (int w = 0; w < WPT; w++) {
            local_a[local_j + w * RTS][local_i] = a[(j + w * RTS) * K + tileK * TILE_SIZE_2 + local_i];
            local_b[local_j + w * RTS][local_i] = b[(tileK * TILE_SIZE_2 + local_j + w * RTS) * N + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE_2; k++) {
            for (int w = 0; w < WPT; w++) {
                sum[w] += local_a[local_j + w * RTS][k] * local_b[k][local_i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < WPT; w++) {
        c[(j + w * RTS) * N + i] = sum[w];
    }
}
