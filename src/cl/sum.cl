#define WORK_GROUP_SIZE 128

__kernel void atomic_sum(__global const unsigned int* xs, unsigned int n,
                         unsigned int batch_size,
                         __global unsigned int *res) {
    int id = get_global_id(0);
    if (id >= n)
        return;
    atomic_add(res, xs[id]);
}

__kernel void cycle_sum(__global const unsigned int* xs, unsigned int n,
                        unsigned int batch_size,
                        __global unsigned int *res) {
    int id = get_global_id(0);
    if (id >= n)
        return;

    int sum = 0;
    for (int i = id * batch_size;
             i < (id + 1) * batch_size && i < n;
             ++i) {
        sum += xs[i];
    }
    atomic_add(res, sum);
}

__kernel void coalesced_cycle_sum(__global const unsigned int* xs, unsigned int n,
                                  unsigned int batch_size,
                                  __global unsigned int *res) {
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);

    int sum = 0;
    for (int i = group_id * group_size * batch_size + local_id;
            i < (group_id + 1) * group_size * batch_size + local_id && i < n;
            i += group_size) {
        sum += xs[i];
    }
    atomic_add(res, sum);
}

__kernel void local_sum(__global const unsigned int* xs, unsigned int n,
                        unsigned int batch_size,
                        __global unsigned int *res) {
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int global_id = get_global_id(0);

    __local int local_xs[/* local_size */ WORK_GROUP_SIZE];
    local_xs[local_id] = global_id < n ? xs[global_id] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        int sum = 0;
        for (int i = 0; i < local_size; ++i) {
            sum += local_xs[i];
        }
        atomic_add(res, sum);
    }
}

__kernel void recursive_sum(__global const unsigned int* xs, unsigned int n,
                            unsigned int batch_size,
                            __global unsigned int *res) {
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int global_id = get_global_id(0);

    __local int local_xs[/*local_size*/ WORK_GROUP_SIZE];
    local_xs[local_id] = global_id < n ? xs[global_id] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nvalues = local_size; nvalues > 1; nvalues /= 2) {
        if (2 * local_id < nvalues) {
            int a = local_xs[local_id];
            int b = local_xs[local_id + nvalues / 2];
            local_xs[local_id] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        atomic_add(res, local_xs[0]);
    }
}