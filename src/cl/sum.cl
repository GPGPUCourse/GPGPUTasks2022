#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 128
#endif

#ifndef VALUES_PER_WORK_ITEM
#define VALUES_PER_WORK_ITEM 64
#endif

#line 6

__kernel void baseline_sum(
    __global const unsigned int *source,
    __global unsigned int *result,
    unsigned int n
) {
    const int globalId = get_global_id(0);

    if (globalId < n) {
        atomic_add(result, source[globalId]);
    }
}

__kernel void cycle_sum(
    __global const unsigned int *source,
    __global unsigned int *result,
    unsigned int n
){
    const int globalId = get_global_id(0);

    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; i++) {
        unsigned int idx = globalId * VALUES_PER_WORK_ITEM + i;
        if (idx < n) {
            res += source[idx];
        }
    }
    atomic_add(result, res);
}

__kernel void coalesced_cycle_sum(
    __global const unsigned int *source,
    __global unsigned int *result,
    unsigned int n
){
    const int localId = get_local_id(0);
    const int groupId = get_group_id(0);
    const int groupSize = get_local_size(0);

    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; i++) {
        unsigned int idx = groupId * groupSize * VALUES_PER_WORK_ITEM + i * groupSize + localId;
        if (idx < n) {
            res += source[idx];
        }
    }
    atomic_add(result, res);
}

__kernel void local_mem_sum(
    __global const unsigned int *source,
    __global unsigned int *result,
    unsigned int n
){
    const int localId = get_local_id(0);
    const int globalId = get_global_id(0);

    __local unsigned int uploaded[WORK_GROUP_SIZE];
    if (globalId >= n)
        uploaded[localId] = 0;
    else
        uploaded[localId] = source[globalId];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0) {
        unsigned int res = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; i++)
            res += uploaded[i];
        atomic_add(result, res);
    }
}

__kernel void tree_sum(
    __global const unsigned int *source,
    __global unsigned int *result,
    unsigned int n) {
    const int localId = get_local_id(0);
    const int globalId = get_global_id(0);

    __local unsigned int uploaded[WORK_GROUP_SIZE];
    if (globalId >= n)
        uploaded[localId] = 0;
    else
        uploaded[localId] = source[globalId];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int len = WORK_GROUP_SIZE / 2; len > 0; len /= 2) {
        if (localId < len) {
            unsigned int a = uploaded[localId];
            unsigned int b = uploaded[localId + len];
            uploaded[localId] = a + b;
        }
        if (len > WARP_SIZE)
            barrier(CLK_LOCAL_MEM_FENCE);
        else if (2 * localId >= len)
            return;
    }

    if (localId == 0)
        atomic_add(result, uploaded[0]);
}