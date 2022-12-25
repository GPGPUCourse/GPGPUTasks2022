#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void prefix_sum(__global const unsigned int* a,
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

__kernel void prefix_sum_other(__global const unsigned int* a,
                             __global unsigned int* c,
                             unsigned int n)
{
    const unsigned int index = get_global_id(0);
    if (index >= n)
        return;

    c[index] = a[2 * index] + a[2 * index + 1];
}