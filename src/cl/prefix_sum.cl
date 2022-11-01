
#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void prefix_sum(__global unsigned int *as, __global unsigned int *bs, unsigned int byte_num) {
    unsigned int i = get_global_id(0);
    if (((1 << byte_num) & (i + 1)) != 0) {
        bs[i] += as[((i + 1) >> byte_num) - 1];
    }
}

__kernel void prefix_sum_reduce(__global unsigned int *as, __global unsigned int *bs)
{
    unsigned int i = get_global_id(0);
    bs[i] = as[i * 2] + as[i * 2 + 1];
}