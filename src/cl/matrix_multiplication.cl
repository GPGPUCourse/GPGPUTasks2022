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

#define WPT ((TILE_SIZE * TILE_SIZE + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE)
#define RTS ((TILE_SIZE + WPT - 1) / WPT)

__kernel void matrix_multiplication_local_mem(
    __global const float *lhs, // M * K
    __global const float *rhs, // K * N
    __global float* result,    // M * N
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    const int local_id = get_local_id(0);
    const int group_id = get_group_id(0);

    const int groups_by_n = (N + TILE_SIZE - 1) / TILE_SIZE;
    const int group_n = group_id % groups_by_n;
    const int group_m = group_id / groups_by_n;
    const int offset_n = group_n * TILE_SIZE;
    const int offset_m = group_m * TILE_SIZE;

    __local float uploaded_lhs[TILE_SIZE * TILE_SIZE];
    __local float uploaded_rhs[TILE_SIZE * (TILE_SIZE + 1)];
    float item_result[WPT];

    for (int i = 0; local_id + i * WORK_GROUP_SIZE < TILE_SIZE * TILE_SIZE; ++i) {
        item_result[i] = 0;
    }

    for (int tile_k = 0; tile_k * TILE_SIZE < K; ++tile_k) {
        const int offset_k = tile_k * TILE_SIZE;

        // uploading
        for (int i = local_id; i < TILE_SIZE * TILE_SIZE; i += WORK_GROUP_SIZE) {
            int k = i % TILE_SIZE;
            int m = i / TILE_SIZE;
            int to = k + m * TILE_SIZE;

            int global_k = offset_k + k;
            int global_m = offset_m + m;
            int from = global_k + global_m * K;

            if (global_k < K && global_m < M) {
                uploaded_lhs[to] = lhs[from];
            } else {
                uploaded_lhs[to] = 0;
            }
        }

        for (int i = local_id; i < TILE_SIZE * TILE_SIZE; i += WORK_GROUP_SIZE) {
            int n = i % TILE_SIZE;
            int k = i / TILE_SIZE;
            int to = n + k * (TILE_SIZE + 1);

            int global_n = offset_n + n;
            int global_k = offset_k + k;
            int from = global_n + global_k * N;

            if (global_k < K && global_n < N) {
                uploaded_rhs[to] = rhs[from];
            } else {
                uploaded_rhs[to] = 0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // multiply tiles
        for (int i = 0; local_id + i * WORK_GROUP_SIZE < TILE_SIZE * TILE_SIZE; ++i) {
            int num = local_id + i * WORK_GROUP_SIZE;
            int n = num % TILE_SIZE;
            int m = num / TILE_SIZE;

            for (int k = 0; k < TILE_SIZE; ++k) {
                int lhs_pos = k + m * TILE_SIZE;
                int rhs_pos = n + k * (TILE_SIZE + 1);

                item_result[i] += uploaded_lhs[lhs_pos] * uploaded_rhs[rhs_pos];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; local_id + i * WORK_GROUP_SIZE < TILE_SIZE * TILE_SIZE; ++i) {
        int num = local_id + i * WORK_GROUP_SIZE;
        int n = num % TILE_SIZE;
        int m = num / TILE_SIZE;

        int global_n = offset_n + n;
        int global_m = offset_m + m;
        int to = global_n + global_m * N;

        if (global_n < N && global_m < M) {
            result[to] = item_result[i];
        }
    }

}


__kernel void matrix_multiplication_fma(
    __global const float *lhs, // M * K
    __global const float *rhs, // K * N
    __global float* result,    // M * N
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    const int col = get_local_id(0);
    const int row = get_local_id(1);

    const int offset_n = get_group_id(0) * TILE_SIZE;
    const int offset_m = get_group_id(1) * TILE_SIZE;

    const int m = row;
    const int global_m = offset_m + m;

    __local float uploaded_lhs[TILE_SIZE * (TILE_SIZE + RTS)];
    __local float uploaded_rhs[TILE_SIZE * (TILE_SIZE + RTS)];
    float item_result[WPT];

    for (int i = 0; i < WPT; ++i) {
        item_result[i] = 0;
    }

    for (int tile_k = 0; tile_k * TILE_SIZE < K; ++tile_k) {
        const int offset_k = tile_k * TILE_SIZE;

        // uploading
        for (int i = 0; i < WPT; ++i) {
            int k = col + i * RTS;
            int to = k + m * (TILE_SIZE + RTS);

            int global_k = offset_k + k;
            int from = global_k + global_m * K;

            if (global_k < K && global_m < M) {
                uploaded_lhs[to] = lhs[from];
            } else {
                uploaded_lhs[to] = 0;
            }
        }

        for (int i = 0; i < WPT; ++i) {
            int n = col + i * RTS;
            int k = row;
            int to = n + k * (TILE_SIZE + RTS);

            int global_n = offset_n + n;
            int global_k = offset_k + k;
            int from = global_n + global_k * N;

            if (global_k < K && global_n < N) {
                uploaded_rhs[to] = rhs[from];
            } else {
                uploaded_rhs[to] = 0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // multiply tiles
        for (int k = 0; k < TILE_SIZE; ++k) {
            float tmp = uploaded_lhs[k + m * (TILE_SIZE + RTS)];
            for (int i = 0; i < WPT; ++i) {
                item_result[i] += tmp * uploaded_rhs[(col + i * RTS) + k * (TILE_SIZE + RTS)];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < WPT; ++i) {
        int n = col + i * RTS;

        int global_n = offset_n + n;
        int to = global_n + global_m * N;

        if (global_n < N && global_m < M) {
            result[to] = item_result[i];
        }
    }

}
