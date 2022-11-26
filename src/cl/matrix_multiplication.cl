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
    for (int tileK = 0; tileK < K / TILE_SIZE; tileK++) {
        tileA[local_j][local_i] = as[j * K + (tileK * TILE_SIZE + local_i)];
        tileB[local_j][local_i] = bs[(tileK * TILE_SIZE + local_j) * N + i];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[local_j][k] * tileB[k][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    cs[j * N + i] = sum;
}

#define WPT 4
#define RTS 4
#define TS 16
__kernel void matrix_multiplication_fma(__global const float* bs,
                                    __global const float* as,
                                    __global float* cs,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N)
{
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    int i = TS * get_group_id(0) + local_i;
    int j = TS * get_group_id(1) + local_j;
//    for (int w = 0; w < WPT; w++) {
//        cs[(j + w * RTS) * M + i] = 0;
//    }
//    barrier(CLK_GLOBAL_MEM_FENCE);

    __local float tileA[TS][TS];
    __local float tileB[TS][TS];

    float sum[WPT];
    for (int w = 0; w < WPT; w++) {
        sum[w] = 0.0f;
    }

    for (int t = 0; t < K / TS; t++) {
        for (int w = 0; w < WPT; w++) {
            const int tiled_i = TS * t + local_i;
            const int tiled_j = TS * t + local_j;
            tileA[local_j + w * RTS][local_i] = as[(tiled_j + w * RTS) * M + i];
            tileB[local_j + w * RTS][local_i] = bs[(j + w * RTS) * K + tiled_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            float tmp = tileA[k][local_i];
            for (int w = 0; w < WPT; w++) {
                sum[w] += tmp * tileB[local_j + w * RTS][k];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < WPT; w++) {
        cs[(j + w * RTS) * M + i] = sum[w];
    }

}