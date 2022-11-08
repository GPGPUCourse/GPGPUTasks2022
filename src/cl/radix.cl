#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void get_zeros_and_ones(__global unsigned int *as, unsigned int n, unsigned int digit,
                                 __global unsigned int *zeros, __global unsigned int *ones) {
    unsigned int i = get_global_id(0);
    if (i >= n)
        return;
    unsigned int tmp = (as[i] >> digit) & 1;
    ones[i] = tmp;
    zeros[i] = 1 - tmp;
}

__kernel void prefix_sum(__global unsigned int *as, __global unsigned int *bs, unsigned int n, unsigned int shift,
                         __global unsigned int *last, unsigned int change_last) {
    unsigned int i = get_global_id(0);
    if (i >= n)
        return;
    if (((i + 1) >> shift) & 1)
        bs[i] += as[((i + 1) >> shift) - 1];
    if (change_last != 0 && i == n - 1)
        last[0] = bs[i];
}

__kernel void pairwise_sum(__global unsigned int *as, __global unsigned int *bs, unsigned int n) {
    unsigned int i = get_global_id(0);
    if (i >= n)
        return;
    bs[i] = as[2 * i] + as[2 * i + 1];
}

__kernel void radix(__global unsigned int *as, __global unsigned int *bs,
                    __global unsigned int *prefix_zeros, __global unsigned int *prefix_ones,
                    unsigned int n, unsigned int digit, __global unsigned int *last) {
    unsigned int i = get_global_id(0);
    if (i >= n)
        return;
    if ((as[i] >> digit) & 1)
        bs[last[0] + prefix_ones[i] - 1] = as[i];
    else
        bs[prefix_zeros[i] - 1] = as[i];
}
