__kernel void atomic_sum(__global const unsigned int* as, unsigned int n, __global int* res) {
    int id = get_global_id(0);
    if (id >= n) {
        return;
    }
    atomic_add(res, as[id]);
}


#define VALUES_PER_WORK_ITEM 64
__kernel void iter_sum(__global const unsigned int* as, unsigned int n, __global int* res) {
    size_t id = get_global_id(0);

    unsigned int sum = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; i++) {
        if (id * VALUES_PER_WORK_ITEM + i >= n) {
            break;
        }
        sum += as[id * VALUES_PER_WORK_ITEM + i];
    }

    atomic_add(res, sum);
}


__kernel void coalesced_sum(__global const unsigned int* as, unsigned int n, __global int* res) {
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);

    unsigned int sum = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; i++) {
        if (group_id * group_size * VALUES_PER_WORK_ITEM + i * group_size + local_id >= n) {
            break;
        }
        sum += as[group_id * group_size * VALUES_PER_WORK_ITEM + i * group_size + local_id];
    }

    atomic_add(res, sum);
}


#define WORK_GROUP_SIZE 128
__kernel void local_sum(__global const unsigned int* as, unsigned int n, __global int* res) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);

    __local int local_as[WORK_GROUP_SIZE];
    local_as[local_id] = as[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        unsigned int sum = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; i++) {
            sum += local_as[i];
        }
        atomic_add(res, sum);
    }
}


__kernel void tree_sum(__global const unsigned int* as, unsigned int n, __global int* res) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);

    __local int local_as[WORK_GROUP_SIZE];
    local_as[local_id] = as[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (2 * local_id < nvalues) {
            int a = local_as[local_id];
            int b = local_as[local_id + nvalues / 2];
            local_as[local_id] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        atomic_add(res, local_as[0]);
    }
}