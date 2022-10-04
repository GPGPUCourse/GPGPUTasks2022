#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256
#define DATA_PER_ITEM 64

__kernel void sum_1(__global const unsigned int *data, int n, __global unsigned int *result) {
    unsigned int i = get_global_id(0);

    if (i >= n) {
        return;
    }

    atomic_add(result, data[i]);
}

__kernel void sum_2(__global const unsigned int *data, int n, __global unsigned int *result) {
    unsigned int i = get_global_id(0);

    if (i * DATA_PER_ITEM >= n) {
        return;
    }

    unsigned int sum = 0;
    for (int j = 0; j < DATA_PER_ITEM; ++j) {
        unsigned int new_i = i * DATA_PER_ITEM + j;
        if (new_i >= n) {
            break;
        }
        sum += data[new_i];
    }

    atomic_add(result, sum);
}

__kernel void sum_3(__global const unsigned int *data, int n, __global unsigned int *result) {
    unsigned int i = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int group_size = get_local_size(0);

    if (group_id * group_size * DATA_PER_ITEM + i >= n) {
      return;
    }

    unsigned int sum = 0;
    for (int j = 0; j < DATA_PER_ITEM; ++j) {
        unsigned int new_i = group_id * group_size * DATA_PER_ITEM + i + j * group_size;
        if (new_i >= n) {
            break;
        }
        sum += data[new_i];
    }

    atomic_add(result, sum);
}

__kernel void sum_4(__global const unsigned int *data, int n, __global unsigned int *result) {
    unsigned int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);
    if (global_id >= n) {
        return;
    }

    __local unsigned int local_data[WORK_GROUP_SIZE];
    local_data[local_id] = data[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        unsigned int sum = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            sum += local_data[i];
        }

        atomic_add(result, sum);
    }
}

__kernel void sum_5(__global const unsigned int *data, int n, __global unsigned int *result) {
    unsigned int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);
    if (global_id >= n) {
        return;
    }

    __local unsigned int local_data[WORK_GROUP_SIZE];
    local_data[local_id] = data[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = WORK_GROUP_SIZE; i > 1; i /= 2) {
        if (2 * local_id < i) {
            unsigned int a = local_data[local_id];
            unsigned int b = local_data[local_id + i / 2];
            local_data[local_id] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
      atomic_add(result, local_data[0]);
    }
}
