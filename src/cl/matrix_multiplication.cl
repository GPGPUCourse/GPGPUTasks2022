#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP 16

__kernel void matrix_multiplication(__global const float *a,
                                    __global const float *b,
                                    __global float *c,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N)
{
    unsigned int local_m = get_local_id(1);
    unsigned int local_k = get_local_id(0);

    unsigned int global_m = get_global_id(1);
    unsigned int global_k = get_global_id(0);

    if (global_m >= M || global_k >= N) {
        return;
    }

    __local float a_local[WORK_GROUP][WORK_GROUP];
    __local float b_local[WORK_GROUP][WORK_GROUP];

    float result = 0.0;
    for (int i = 0; i < K; i += WORK_GROUP) {
        a_local[local_m][local_k] = a[global_m * K + local_k + i];
        b_local[local_m][local_k] = b[N * (local_m + i) + global_k];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < WORK_GROUP; ++j) {
            result += a_local[local_m][j] * b_local[j][local_k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[global_m * N + global_k] = result;
}