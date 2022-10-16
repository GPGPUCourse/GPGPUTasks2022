#define TILE_SIZE 16
__kernel void matrix_multiplication_1(__global const float *a, __global const float *b, __global float *c,
                                      unsigned int M, unsigned int N, unsigned int K)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int tileK = 0; tileK * TILE_SIZE < K; tileK++) {
        tileA[local_j][local_i] = a[j * K + TILE_SIZE * tileK + local_i];
        tileB[local_j][local_i] = b[(TILE_SIZE * tileK + local_j) * N + i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[local_j][k] * tileB[k][local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[j * N + i] = sum;
}


#define TILE_SIZE 16
#define THREAD_WORK 4
#define RTS TILE_SIZE / THREAD_WORK
__kernel void matrix_multiplication_2(__global const float *a, __global const float *b, __global float *c,
                                      unsigned int M, unsigned int N, unsigned int K)
{
    const int offset_x = get_group_id(0) * TILE_SIZE;
    const int offset_y = get_group_id(1) * TILE_SIZE;

    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);

    const int i = offset_x + local_i;
    const int j = offset_y + local_j;

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum[THREAD_WORK];
    for (int w = 0; w < THREAD_WORK; w++) {
        sum[w] = 0.0f;
    }

    for (int tileK = 0; tileK * TILE_SIZE < K; tileK++) {

        for (int w = 0; w < THREAD_WORK; w++) {
            tileA[local_j + w * RTS][local_i] = a[(j + w * RTS) * K + TILE_SIZE * tileK + local_i];
            tileB[local_j + w * RTS][local_i] = b[(TILE_SIZE * tileK + local_j + w * RTS) * N + i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            const float tmp = tileB[k][local_i];
            for (int w = 0; w < THREAD_WORK; w++) {
                sum[w] += tileA[local_j + w * RTS][k] * tmp;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < THREAD_WORK; w++) {
        c[(j + w * RTS) * N + i] = sum[w];
    }
}