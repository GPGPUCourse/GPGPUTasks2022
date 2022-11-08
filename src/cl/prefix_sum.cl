#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void prefix_sum(__global unsigned int *as, __global unsigned int *bs, unsigned int n, unsigned int shift) {
    unsigned int i = get_global_id(0);
    if (i >= n)
        return;
    if (((i + 1) >> shift) & 1)
        bs[i] += as[((i + 1) >> shift) - 1];
}

__kernel void pairwise_sum(__global unsigned int *as, __global unsigned int *bs, unsigned int n) {
    unsigned int i = get_global_id(0);
    if (i >= n)
        return;
    bs[i] = as[2 * i] + as[2 * i + 1];
}
