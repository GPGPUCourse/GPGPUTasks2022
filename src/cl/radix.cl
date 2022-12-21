#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define SEGMENT_SIZE 128


__kernel void radix_counters(__global const unsigned int *as, __global unsigned int *counters, unsigned int offset, unsigned int n) {
    const unsigned int id = get_global_id(0);
    if (id * SEGMENT_SIZE >= n)
        return;

    unsigned int segment_begin = id * SEGMENT_SIZE;
    unsigned int segment_end = (id + 1) * SEGMENT_SIZE;

    unsigned int counter = 0;

    for (int i = segment_begin; i < segment_end; i++) {
        counter += (as[i] >> offset) & 1;
    }

    counters[id] = counter;
}

__kernel void nullate(__global unsigned int *as, unsigned int n) {
    const unsigned int id = get_global_id(0);
    if (id >= n)
        return;

    as[id] = 0;
}


__kernel void prefix_sum_reduce(__global const unsigned int* a,
                         __global unsigned int* b,
                         unsigned int n,
                         unsigned int level)
{
    const unsigned int index = get_global_id(0);
    if (index >= n)
        return;

    if (((index+1)>>level) & 1) {
        b[index] += a[((index+1)>>level) - 1];
    }

}


__kernel void prefix_sum_gather(__global const unsigned int* a,
                               __global unsigned int* c,
                               unsigned int n)
{
    const unsigned int index = get_global_id(0);
    if (index >= n)
        return;

    c[index] = a[2 * index] + a[2 * index + 1];
}


__kernel void local_sort(__local unsigned int *tmp, __local unsigned int *counter, unsigned int offset, __global unsigned int *as) {
    unsigned int tmp_counter = 0;
    unsigned int tmp2[SEGMENT_SIZE];
    for (unsigned int i = 0; i < SEGMENT_SIZE; i++) {
        tmp2[i] = tmp[i];
        tmp_counter += ((tmp2[i] >> offset) & 1);
    }

    unsigned int i = 0, j = SEGMENT_SIZE - tmp_counter;
    for (unsigned int k = 0; k < SEGMENT_SIZE; k++) {
        if (!((tmp2[k] >> offset) & 1))
            tmp[i++] = tmp2[k];
        else
            tmp[j++] = tmp2[k];
    }
//    for (unsigned int i = 0; i < SEGMENT_SIZE; i++) {
//        for (unsigned int j = i+1; j < SEGMENT_SIZE; j++) {
////            if (tmp[i] == tmp[j])
////                as[1000000] = 10;
//        }
//    }

    *counter = tmp_counter;
}


__kernel void radix_sort(__global const unsigned int *counters, __global unsigned int *as, __global unsigned int *bs, unsigned int offset, unsigned int n) {
    const unsigned int id = get_global_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int local_id = get_local_id(0);
    if (id >= n)
        return;

    unsigned int zero_all_count = n - counters[n / SEGMENT_SIZE - 1];

    __local unsigned int counter_sum;
    __local unsigned int counter;
    __local unsigned int tmp[SEGMENT_SIZE];
    tmp[local_id] = as[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (id == group_id * SEGMENT_SIZE) {
        if (group_id == 0)
            counter_sum = 0;
        else
            counter_sum = counters[group_id-1];

        local_sort(tmp, &counter, offset, as);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int zero_counter = SEGMENT_SIZE - counter;
    unsigned int zero_counter_sum = group_id * SEGMENT_SIZE - counter_sum;


    unsigned int new_pos = (local_id < zero_counter
            ? zero_counter_sum + local_id
            : zero_all_count + counter_sum + (local_id - zero_counter));

    barrier(CLK_GLOBAL_MEM_FENCE);
    bs[new_pos] = tmp[local_id];

}
