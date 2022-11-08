#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP 256

__kernel void bitonic_local(__global float *as, int n, int size, int interval) {
    unsigned int i = get_global_id(0);
    unsigned int local_i = get_local_id(0);
    __local float local_as[WORK_GROUP];

    if (i < n) {
        local_as[local_i] = as[i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int isChange = i % (2 * size) < size;
    for (int shift = interval; shift > 0; shift /= 2) {
        if (i + shift < n && i % (2 * shift) < shift) {
            float a = local_as[local_i];
            float b = local_as[local_i + shift];
            if ((a > b) == isChange) {
                local_as[local_i] = b;
                local_as[local_i + shift] = a;
            }
        }

      barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < n) {
        as[i] = local_as[local_i];
    }
}

__kernel void bitonic(__global float *as, int n, int size, int interval) {
    unsigned int i = get_global_id(0);

    bool isChange = i % (2 * size) < size;
    if (i + interval < n && i % (2 * interval) < interval) {
        float a = as[i];
        float b = as[i + interval];
        if ((a > b) == isChange) {
            as[i] = b;
            as[i + interval] = a;
        }
    }
}
