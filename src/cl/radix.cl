
#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128
#define MAX_NUM (1 << 4) // = (1 << byte_cnt)

__kernel void radix_count(__global unsigned int *as, unsigned int N, __global unsigned int *bs, unsigned int byte_num, unsigned int byte_cnt) {
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    __local unsigned int storage[WORK_GROUP_SIZE];
    __local unsigned int res[MAX_NUM];

    storage[local_id] = as[global_id];
    if (local_id < (1 << byte_cnt)) {
        res[local_id] = 0;
    }


    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        for (unsigned int i = 0; i < WORK_GROUP_SIZE; i++) {
            unsigned int j = (storage[i] >> byte_num) & ((1 << byte_cnt) - 1);
            res[j] += 1;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < (1 << byte_cnt)) {
        bs[local_id * (N / WORK_GROUP_SIZE) + global_id / WORK_GROUP_SIZE] = res[local_id];
    }
}

__kernel void radix_sort(__global unsigned int *as, unsigned int N, __global unsigned int *prefix, __global unsigned int *res, unsigned int byte_num, unsigned int byte_cnt) {
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int num_prefix = global_id / WORK_GROUP_SIZE;

    __local unsigned int storage[WORK_GROUP_SIZE];
    __local unsigned int cnts[MAX_NUM];
    __local unsigned int prev_counts[MAX_NUM];
    __local unsigned int local_res[WORK_GROUP_SIZE];

    storage[local_id] = as[global_id];

    if (local_id < (1 << byte_cnt)) {
        if (num_prefix == 0) {
            if (local_id == 0) {
                prev_counts[local_id] = 0;
            } else {
                prev_counts[local_id] = prefix[local_id * (N / WORK_GROUP_SIZE) - 1];
            }
        } else {
            prev_counts[local_id] = prefix[local_id * (N / WORK_GROUP_SIZE) + num_prefix - 1];
        }
        cnts[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    if (local_id == 0) {
        for (unsigned int i = 0; i < WORK_GROUP_SIZE; i++) {
            unsigned int j = (storage[i] >> byte_num) & ((1 << byte_cnt) - 1);
            local_res[i] = prev_counts[j] + cnts[j];
            cnts[j] += 1;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    res[local_res[local_id]] = storage[local_id];
}