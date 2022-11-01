#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void prefix(__global unsigned int *as, __global unsigned int *bs, unsigned int n, unsigned int bit) {
    unsigned int i = get_global_id(0);

    if (i < n && (((i + 1) >> bit) & 1)) {
        bs[i] += as[((i + 1) >> bit) - 1];
    }
}

__kernel void reduce(__global unsigned int *as, __global unsigned int *bs, unsigned int n) {
    unsigned int i = get_global_id(0);
    if (i < n) {
      bs[i] = as[2*i] + as[2*i + 1];
    }
}

__kernel void zero(__global unsigned int *bs,  unsigned int n) {
    unsigned int i = get_global_id(0);
    if (i < n) {
        bs[i] = 0;
    }
}