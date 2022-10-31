#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 128
#endif

__kernel void bitonic(
    __global float *as,
    const unsigned int sorted_block_size,
    const unsigned int step
) {
    const unsigned int global_id = get_global_id(0);

    const unsigned int shift = sorted_block_size >> step;
    const unsigned int block = global_id / shift;
    const unsigned int pos = global_id + block * shift;

    const unsigned int color_block = global_id / sorted_block_size;
    const int sign = 1 - 2 * (color_block % 2);

    float fst = as[pos];
    float snd = as[pos + shift];

    if (sign * fst > sign * snd) {
        as[pos] = snd;
        as[pos + shift] = fst;
    }
}

__kernel void bitonic_local(
    __global float *as,
    const unsigned int sorted_block_size
) {
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int group_id = get_group_id(0);

    const unsigned int upload_shift = group_id * WORK_GROUP_SIZE;

    __local float uploaded[2 * WORK_GROUP_SIZE];

    uploaded[local_id] = as[global_id + upload_shift];
    uploaded[local_id + WORK_GROUP_SIZE] = as[global_id + WORK_GROUP_SIZE + upload_shift];

    barrier(CLK_LOCAL_MEM_FENCE);

    {
        const unsigned int color_block = global_id / sorted_block_size;
        const int sign = 1 - 2 * (color_block % 2);

        for (unsigned int shift = sorted_block_size; shift > 0; shift >>= 1) {
            const unsigned int block = local_id / shift;
            const unsigned int pos = local_id + block * shift;

            float fst = uploaded[pos];
            float snd = uploaded[pos + shift];

            if (sign * fst > sign * snd) {
                uploaded[pos] = snd;
                uploaded[pos + shift] = fst;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    as[global_id + upload_shift] = uploaded[local_id];
    as[global_id + WORK_GROUP_SIZE + upload_shift] = uploaded[local_id + WORK_GROUP_SIZE];
}
