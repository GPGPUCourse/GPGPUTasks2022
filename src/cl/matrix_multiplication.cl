#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define TILE_SIZE 16
__kernel void matrix_multiplication1(__global const float *a, __global const float *b, __global float *c,
                                     unsigned int M, unsigned int K, unsigned int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        tileA[local_j][local_i] = a[j * K + tileK * TILE_SIZE + local_i];
        tileB[local_j][local_i] = b[(local_j + TILE_SIZE * tileK) * N + i];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[local_j][k] * tileB[k][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (i < N && j < M) {
            c[j * N + i] = sum;
        }
    }
}

#define THREAD_WORK 4
__kernel void matrix_multiplication2(__global const float *a, __global const float *b, __global float *c,
                                     unsigned int M, unsigned int K, unsigned int N) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE][TILE_SIZE + 1];

    float sum[THREAD_WORK];
    for (int w = 0; w < THREAD_WORK; w++) {
        sum[w] = 0;
    }
    for (int tileK = 0; tileK * TILE_SIZE < K; tileK++) {
        for (int w = 0; w < THREAD_WORK; w++) {
            tileA[local_j][local_i * THREAD_WORK + w] = a[j * K + TILE_SIZE * tileK + local_i * THREAD_WORK + w];
            tileB[local_j][local_i * THREAD_WORK + w] = b[(TILE_SIZE * tileK + local_j) * N + i * THREAD_WORK + w];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            float tmp = tileA[local_j][k];
            for (int w = 0; w < THREAD_WORK; w++) {
                sum[w] += tmp * tileB[k][local_i * THREAD_WORK + w];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < THREAD_WORK; w++) {
        c[j * N + i * THREAD_WORK + w] = sum[w];
    }
}