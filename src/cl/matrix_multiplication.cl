#define TILE_SIZE 16

__kernel void matrix_multiplication_1(__global float *a, __global float *b, __global float *c,
                                    unsigned int M, unsigned int K, unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    __local float tileA[TILE_SIZE][TILE_SIZE + 1]; // one element in every row for fake element which fix bank-conflicts
    __local float tileB[TILE_SIZE][TILE_SIZE + 1]; // one element in every row for fake element which fix bank-conflicts

    float sum = 0.0f;
    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        if (j < M && i < N) {
            tileA[local_j][local_i] = a[j * K + (tileK * TILE_SIZE + local_i)];
            tileB[local_j][local_i] = b[(tileK * TILE_SIZE + local_j) * N + i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (j < M && i < N) {
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += tileA[local_j][k] * tileB[k][local_i];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (j < M && i < N) {
        c[j * N + i] = sum;
    }
}

#define THREAD_WORK 4
__kernel void matrix_multiplication_2(__global float *a, __global float *b, __global float *c,
                                      unsigned int M, unsigned int K, unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    __local float tileA[TILE_SIZE][TILE_SIZE + 1]; // one element in every row for fake element which fix bank-conflicts
    __local float tileB[TILE_SIZE][TILE_SIZE + 1]; // one element in every row for fake element which fix bank-conflicts

    float sum[THREAD_WORK];
    for (int th = 0; th < THREAD_WORK; ++th) {
        sum[th] = 0.0f;
    }
    int STRIPE_SIZE = TILE_SIZE / THREAD_WORK;
    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        if (j < M && i < N / THREAD_WORK) {
            for (int th = 0; th < THREAD_WORK; ++th) {
                tileA[local_j][local_i + th * STRIPE_SIZE] = a[j * K + (tileK * TILE_SIZE + local_i) +
                                                               th * STRIPE_SIZE];
                tileB[local_j][local_i + th * STRIPE_SIZE] = b[(tileK * TILE_SIZE + local_j) * N + i +
                                                               th * STRIPE_SIZE];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (j < M && i < N / THREAD_WORK) {
            for (int k = 0; k < TILE_SIZE; ++k) {
                float tmp = tileA[local_j][k];
                for (int th = 0; th < THREAD_WORK; ++th) {
                    sum[th] += tmp * tileB[k][local_i + th * STRIPE_SIZE];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (j < M && i < N / THREAD_WORK) {
        for (int th = 0; th < THREAD_WORK; ++th) {
            c[j * N + i + th * STRIPE_SIZE] = sum[th];
        }
    }
}