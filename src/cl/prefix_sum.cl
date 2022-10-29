#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

__kernel void prefix_sum(__global unsigned int *cur_prefix_sum, __global unsigned int *cur_part_sum,
                         unsigned int cur_bit, unsigned int n) {
    unsigned int global_i = get_global_id(0);
    if (global_i >= n)
        return;
    if (((global_i + 1) >> cur_bit) & 1) {
        unsigned int part_sum_id = ((global_i + 1) >> cur_bit) - 1;
        cur_prefix_sum[global_i] += cur_part_sum[part_sum_id];
    }
}

__kernel void part_sum(__global unsigned int *cur_part_sum, __global unsigned int *next_part_sum,
                       unsigned int numOfNextParts) {
    unsigned int global_i = get_global_id(0);
    if (global_i >= numOfNextParts)
        return;
    next_part_sum[global_i] = cur_part_sum[global_i * 2] + cur_part_sum[global_i * 2 + 1];
}