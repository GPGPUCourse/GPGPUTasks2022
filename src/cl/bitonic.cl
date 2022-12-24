#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

void swap_global(__global float* a, __global float* b) {
    unsigned int tmp = *a;
    *a = *b;
    *b = tmp;
}

void sort_global(__global float* a, __global float* b) {
    if (*a > *b)
        swap_global(a, b);
}

void rev_sort_global(__global float* a, __global float* b) {
    if (*a < *b)
        swap_global(a, b);
}

__kernel void bitonic_big_segment(__global float *as, unsigned int segment_size, unsigned int size, bool reversable, unsigned int n) {
    const unsigned int index = get_global_id(0);
    if (index >= n)
        return;

    const unsigned int i1 = index % (size / 2) + size * (index / size * 2);
    const unsigned int i2 = i1 + size/2;

    bool order = i1 / segment_size % 2;

    if (order)


}
