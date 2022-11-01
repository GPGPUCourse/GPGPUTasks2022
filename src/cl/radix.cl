#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 128
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 4
#endif

#define BATCH_COUNT (1 << BATCH_SIZE)
#define MASK (BATCH_COUNT - 1)

#define get_type(x, bit) ((x >> bit) & MASK)

#define get_groups(x) ((x + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE)

__kernel void count(
    __global unsigned int *as,
    __global unsigned int *hist,
    unsigned int n,
    unsigned int bit
) {
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int groups = get_groups(n);

    __local unsigned int local_hist[BATCH_COUNT * WORK_GROUP_SIZE];
    for (unsigned int i = 0; i < BATCH_COUNT; ++i) {
        local_hist[i * WORK_GROUP_SIZE + local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id < n) {
        unsigned int type = get_type(as[global_id], bit);
        local_hist[local_id * BATCH_COUNT + type] = 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < BATCH_COUNT) {
        unsigned int sum = 0;
        for (unsigned int i = 0; i < WORK_GROUP_SIZE; ++i) {
            sum += local_hist[i * BATCH_COUNT + local_id];
        }

        hist[local_id * groups + group_id] = sum;
    }

}

__kernel void radix(
    __global unsigned int *as,
    __global unsigned int *bs,
    unsigned int n,
    __global unsigned int *hist,
    unsigned int bit
) {
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int groups = get_groups(n);

    __local unsigned int uploaded[WORK_GROUP_SIZE];
    __local unsigned int local_hist[BATCH_COUNT * WORK_GROUP_SIZE];

    for (unsigned int i = 0; i < BATCH_COUNT; ++i) {
        local_hist[i * WORK_GROUP_SIZE + local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id < n) {
        unsigned int x = as[global_id];
        unsigned int type = get_type(x, bit);
        uploaded[local_id] = x;
        local_hist[type * WORK_GROUP_SIZE + local_id] = 1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < BATCH_COUNT) {
        unsigned int sum = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            unsigned int idx = local_id * WORK_GROUP_SIZE + i;
            unsigned int tmp = local_hist[idx];
            local_hist[idx] = sum;
            sum += tmp;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id < n) {
        unsigned int x = uploaded[local_id];
        unsigned int type = get_type(x, bit);
        unsigned int hist_idx = type * groups + group_id;
        unsigned int new_idx = (hist_idx == 0 ? 0 : hist[hist_idx - 1]) + local_hist[type * WORK_GROUP_SIZE + local_id];
        bs[new_idx] = x;
    }

}
