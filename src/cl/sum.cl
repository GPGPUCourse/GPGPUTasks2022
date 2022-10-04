#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALUES_PER_WORK_ITEM 64
#define WORK_GROUP_SIZE 128

__kernel void sum_gpu_1(__global const int *xs, int n, __global int *res) {
    int id = get_global_id(0);
    if (id >= n)
        return;
    atomic_add(res, xs[id]);
}

__kernel void sum_gpu_2(__global const int *xs, int n, __global int *res) {
    int id = get_global_id(0);

    int sum = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        unsigned int index = id * VALUES_PER_WORK_ITEM + i;
        if (index < n)
            sum += xs[index];
    }
    atomic_add(res, sum);
}

__kernel void sum_gpu_3(__global const int *xs, int n, __global int *res) {
    int localId = get_local_id(0);
    int groupId = get_group_id(0);
    int groupSize = get_local_size(0);

    int sum = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        unsigned int index = groupId * groupSize * VALUES_PER_WORK_ITEM + i * groupSize + localId;
        if (index < n)
            sum += xs[index];
    }
    atomic_add(res, sum);
}

__kernel void sum_gpu_4(__global const int *xs, int n, __global int *res) {
    int localId = get_local_id(0);
    int globalId = get_global_id(0);

    __local int local_xs[WORK_GROUP_SIZE];
    local_xs[localId] = globalId < n ? xs[globalId] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId == 0) {
        int sum = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            sum += local_xs[i];
        }
        atomic_add(res, sum);
    }
}

__kernel void sum_gpu_5(__global const int *xs, int n, __global int *res) {
    int localId = get_local_id(0);
    int globalId = get_global_id(0);

    __local int local_xs[WORK_GROUP_SIZE];
    local_xs[localId] = globalId < n ? xs[globalId] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (2 * localId < nvalues) {
            int a = local_xs[localId];
            int b = local_xs[localId + nvalues/2];
            local_xs[localId] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localId == 0) {
        atomic_add(res, local_xs[0]);
    }
}
