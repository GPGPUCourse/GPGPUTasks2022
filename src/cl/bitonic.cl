#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void bitonic(__global float *as, int n, int m, int k) {
    int i = get_global_id(0);
    if (i % (2 * k) >= k || i + k >= n)
        return;
    bool inc = as[i] < as[i + k], even = (i / m) % 2 == 0;
    if (even && !inc || !even && inc) {
        float tmp = as[i];
        as[i] = as[i + k];
        as[i + k] = tmp;
    }
}

#define GROUP_SIZE 256

__kernel void bitonic_local(__global float *as, int n, int m, int k) {
    int i = get_global_id(0);
    bool even = (i / m) % 2 == 0;

    int local_i = get_local_id(0);
    __local float local_as[GROUP_SIZE];
    local_as[local_i] = as[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (; k >= 1; k /= 2) {
        bool inc = local_as[local_i] < local_as[local_i + k];
        if (i + k < n && i % (2 * k) < k && (even && !inc || !even && inc)) {
            float tmp = local_as[local_i];
            local_as[local_i] = local_as[local_i + k];
            local_as[local_i + k] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    as[i] = local_as[local_i];
}
