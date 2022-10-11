#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 16
__kernel void matrix_multiplication(__global const float* as,
                                    __global const float* bs,
                                    __global float* cs,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0;
    for (int tileK = 0; tileK * TILE_SIZE < K; tileK++) {
        tileA[local_j][local_i] = as[j * K + (tileK * TILE_SIZE + local_i)];
        tileA[local_j][local_i] = bs[(tileK * TILE_SIZE + local_j) * N + i];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[local_j][k] * tileB[k][local_i];
        }
    }
    cs[j * N + i] = sum;
}

#define THREAD_WORK 4
#define RTS 4
__kernel void matrix_multiplication_fma(__global const float* as,
                                    __global const float* bs,
                                    __global float* cs,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum[THREAD_WORK];
    for (int w = 0; w < THREAD_WORK; w++) {
        sum[w] = 0;
    }

    for (int tileK = 0; tileK * TILE_SIZE < K; tileK++) {
        for (int w = 0; w < THREAD_WORK && local_j + w * RTS < TILE_SIZE && (j + w * RTS) < K && (tileK * TILE_SIZE + local_j + w * RTS) < N; w++) {
            tileA[local_j + w * RTS][local_i] = as[(j + w * RTS) * M + (tileK * TILE_SIZE + local_i)];
            tileB[local_j + w * RTS][local_i] = bs[(tileK * TILE_SIZE + local_j + w * RTS) * K + i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            float tmp = tileA[local_j][k];
            for (int w = 0; w < THREAD_WORK && local_i*THREAD_WORK+w*RTS < N; w++) {
                sum[w] += tmp * tileB[k][local_i+w*RTS];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < THREAD_WORK && (j + w * RTS) * N + i < M * N; w++) {
        cs[(j + w * RTS) * N + i] = sum[w];
    }

}