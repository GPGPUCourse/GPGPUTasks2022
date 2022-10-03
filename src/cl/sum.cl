#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALUES_PER_WORK_ITEM 64u
#define WORK_GROUP_SIZE 128u

__kernel void global_atomic_sum(__global const unsigned int* xs, unsigned int n, __global unsigned int *res) {
    int id = get_global_id(0);
    if (id >= n)
        return;
    atomic_add(res, xs[id]);
}

__kernel void cycle_sum(__global const unsigned int* xs, unsigned int n, __global unsigned int *res) {
    int id = get_global_id(0);
    unsigned int sum = 0;
    for (int i = 0, index = id * VALUES_PER_WORK_ITEM; i < VALUES_PER_WORK_ITEM; i++, index++) {
        if (index >= n)
            break;
        sum += xs[index];
    }
    atomic_add(res, sum);
}

__kernel void cycle_coalesced_sum(__global const unsigned int* xs, unsigned int n, __global unsigned int *res) {
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);

    unsigned int sum = 0;
    for (int i = 0, index = group_id * group_size * VALUES_PER_WORK_ITEM + local_id; i < VALUES_PER_WORK_ITEM;
         i++, index += group_size) {
        if (index >= n)
            break;
        sum += xs[index];
    }
    atomic_add(res, sum);
}

__kernel void local_mem_main_thread_sum(__global const unsigned int* xs, unsigned int n, __global unsigned int *res) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);

    __local unsigned int local_xs[WORK_GROUP_SIZE];
    local_xs[local_id] = global_id < n ? xs[global_id] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        int sum = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; i++) {
            sum += local_xs[i];
        }
        atomic_add(res, sum);
    }
}

__kernel void tree_sum(__global const unsigned int* xs, unsigned int n, __global unsigned int *res) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);

    __local unsigned int local_xs[WORK_GROUP_SIZE];
    local_xs[local_id] = global_id < n ? xs[global_id] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (2 * local_id < nvalues) {
            int a = local_xs[local_id];
            int b = local_xs[local_id + nvalues / 2];
            local_xs[local_id] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        atomic_add(res, local_xs[0]);
    }
}
