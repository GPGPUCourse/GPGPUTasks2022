#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

__kernel void init(
    __global unsigned int *as,
    unsigned int n
) {
    unsigned int i = get_global_id(0);
    if (i < n) {
        as[i] = 0;
    }
}

__kernel void update(
    __global unsigned int *as,
    __global unsigned int *bs,
    unsigned int n,
    unsigned int bit
) {
    unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    unsigned int id = i + 1;

    if ((id >> bit) & 1) {
        bs[i] += as[(id >> bit) - 1];
    }
}

__kernel void reduce(
    __global unsigned int *as,
    __global unsigned int *bs,
    unsigned int n
) {
    unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    bs[i] = 0;
    if (i * 2 >= n) {
        return;
    }
    bs[i] += as[i * 2];
    if (i * 2 + 1 >= n) {
        return;
    }
    bs[i] += as[i * 2 + 1];
}
