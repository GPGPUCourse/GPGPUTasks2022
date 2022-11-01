#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void bitonic(__global float *as, int n, int k, int m) {
    int id = get_global_id(0);
    bool odd = (id / m) % 2, inc = as[id] < as[id + k / 2];
    if (id + k / 2 < n && id % k < k / 2 && (odd && inc || !odd && !inc)) {
        float tmp = as[id];
        as[id] = as[id + k / 2];
        as[id + k / 2] = tmp;
    }
}

#define GROUP_SIZE 128

__kernel void bitonic_local(__global float *as, int n, int k, int m) {
    int id = get_global_id(0);
    int local_id = get_local_id(0);
    __local float local_as[GROUP_SIZE];
    local_as[local_id] = as[id];
    barrier(CLK_LOCAL_MEM_FENCE);
    bool odd = (id / m) % 2, inc = as[id] < as[id + k / 2];
    if (id + k / 2 < n && id % k < k / 2 && (odd && inc || !odd && !inc)) {
        float tmp = local_as[local_id];
        local_as[local_id] = local_as[local_id + k / 2];
        local_as[local_id + k / 2] = tmp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    as[id] = local_as[local_id];
}
