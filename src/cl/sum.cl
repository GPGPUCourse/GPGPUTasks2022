#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum_1(const __global unsigned int* a,
                        __global unsigned int* result,
                                 unsigned int n)
{
    const unsigned int index = get_global_id(0);

    if (index > n)
        return;

    atomic_add(result, a[index]);
}

#define VALUES_PER_WORK_ITEM 64
__kernel void sum_2(const __global unsigned int* a,
                    __global unsigned int* result,
                    unsigned int n)
{
    const unsigned int index = get_global_id(0);

    if (index > n / VALUES_PER_WORK_ITEM)
        return;

    unsigned int sum = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; i++) {
        sum += a[index * VALUES_PER_WORK_ITEM + i];
    }

    atomic_add(result, sum);
}

__kernel void sum_3(const __global unsigned int* a,
                    __global unsigned int* result,
                    unsigned int n)
{
    const unsigned int localId = get_local_id(0);
    const unsigned int groupId = get_group_id(0);
    const unsigned int groupSize = get_local_size(0);

    if ((groupId) * groupSize >= n / VALUES_PER_WORK_ITEM)
        return;

    unsigned int sum = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM && groupId * groupSize * VALUES_PER_WORK_ITEM + i * groupSize + localId < n; i++) {
        sum += a[groupId * groupSize * VALUES_PER_WORK_ITEM + i * groupSize + localId];
    }

    atomic_add(result, sum);
}

#define WORK_GROUP_SIZE 256
__kernel void sum_4(const __global unsigned int* a,
                    __global unsigned int* result,
                    unsigned int n)
{
    const unsigned int localId = get_local_id(0);
    const unsigned int globalId = get_global_id(0);

    if (globalId >= n)
        return;

    __local unsigned int local_a[WORK_GROUP_SIZE];
    local_a[localId] = a[globalId];

    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId == 0) {
        unsigned int sum = 0;
        for (int i = 0; i < WORK_GROUP_SIZE && i + globalId < n; i++) {
            sum += local_a[i];
        }

        atomic_add(result, sum);
    }
}

#define WORK_GROUP_SIZE 256
__kernel void sum_5(const __global unsigned int* a,
                    __global unsigned int* result,
                    unsigned int n)
{
    const unsigned int localId = get_local_id(0);
    const unsigned int globalId = get_global_id(0);

    if (globalId >= n)
        return;

    __local unsigned int local_a[WORK_GROUP_SIZE];
    local_a[localId] = a[globalId];

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (2 * localId < nvalues && globalId + nvalues/2 < n) {
            int x = local_a[localId];
            int y = local_a[localId + nvalues/2];
            local_a[localId] = x + y;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localId == 0) {
        atomic_add(result, local_a[0]);
    }
}