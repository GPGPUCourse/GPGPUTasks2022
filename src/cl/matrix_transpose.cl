#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP 16

__kernel void matrix_transpose(__global const float *m, __global float *mT, unsigned int M, unsigned int K)
{
    unsigned int local_m = get_local_id(1);
    unsigned int local_k = get_local_id(0);

    unsigned int global_m = get_global_id(1);
    unsigned int global_k = get_global_id(0);

    if (global_m >= M || global_k >= K) {
        return;
    }

    __local float local_data[WORK_GROUP][WORK_GROUP];

    local_data[local_m][local_k] = m[global_m * K + global_k];

    barrier(CLK_LOCAL_MEM_FENCE);

    mT[global_k * M + global_m] = local_data[local_m][local_k];
}