#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void swap_global(__global float* a, __global float* b) {
    float tmp = *a;
    *a = *b;
    *b = tmp;
}

__kernel void sort_global(__global float* a, __global float* b) {
    if (*a > *b)
        swap_global(a, b);
}

__kernel void rev_sort_global(__global float* a, __global float* b) {
    if (*a < *b)
        swap_global(a, b);
}

__kernel void bitonic_big_segment(__global float *as, unsigned int segment_size, unsigned int size, unsigned int n) {
    const unsigned int index = get_global_id(0);
    if (index % size >= size / 2)
        return;

    const unsigned int i1 = index % (size / 2) + size * (index / size);
    const unsigned int i2 = i1 + size/2;

    bool order = i1 / segment_size % 2;

    if (!order) {
        sort_global(&as[i1], &as[i2]);
    } else {
        rev_sort_global(&as[i1], &as[i2]);
    }
}

__kernel void swap_local(__local float* a, __local float* b) {
    float tmp = *a;
    *a = *b;
    *b = tmp;
}

__kernel void sort_local(__local float* a, __local float* b) {
    if (*a > *b)
        swap_local(a, b);
}

__kernel void rev_sort_local(__local float* a, __local float* b) {
    if (*a < *b)
        swap_local(a, b);
}

#define MAX_LOCAL_SEGMENT_SIZE 128
__kernel void bitonic_small_segment(__global float *as, unsigned int segment_size, unsigned int n) {
    const unsigned int index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);
    if (index >= n)
        return;

    __local float tmp[MAX_LOCAL_SEGMENT_SIZE];
    tmp[local_index] = as[index];
    barrier(CLK_LOCAL_MEM_FENCE);

    bool order = (index / segment_size) % 2;


    for (unsigned int size = segment_size; size >= 2; size /= 2) {
        if (local_index % size < size / 2) {
            const unsigned int i1 = local_index % (size / 2) + size * (local_index / size);
            const unsigned int i2 = i1 + size / 2;
            if (!order) {
                sort_local(&tmp[i1], &tmp[i2]);
            } else {
                rev_sort_local(&tmp[i1], &tmp[i2]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    as[index] = tmp[local_index];
}
