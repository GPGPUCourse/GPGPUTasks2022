#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP 16

__kernel void matrix_transpose(__global const float *m, __global float *mT, unsigned int M, unsigned int K)
{
    unsigned int local_m = get_local_id(1);
    unsigned int local_k = get_local_id(0);

    unsigned int group_m = get_group_id(1);
    unsigned int group_k = get_group_id(0);

    unsigned int global_m = get_global_id(1);
    unsigned int global_k = get_global_id(0);

    __local float local_data[WORK_GROUP][WORK_GROUP + 1];

    local_data[local_m][local_k] = m[global_m * K + global_k];

    barrier(CLK_LOCAL_MEM_FENCE);

    mT[(group_k * WORK_GROUP + local_k) * M + (group_m * WORK_GROUP + local_m)] = local_data[local_m][local_k];
}