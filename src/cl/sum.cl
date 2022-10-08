__kernel void sumGlobalAdd(__global const uint* a, uint n, __global uint* result) {
    size_t i = get_global_id(0);
    if (i >= n)
        return;
    atomic_add(result, a[i]);
}


#define WPT 64

__kernel void sumLoop(__global const uint* a, uint n, __global uint* result) {
    size_t i = get_global_id(0);
    size_t offset = i * WPT;
    if (offset >= n)
        return;

    uint sum = 0;
    for (size_t j = 0; j < WPT && offset + j < n; ++j)
        sum += a[offset + j];

    atomic_add(result, sum);
}

__kernel void sumLoopCoalesced(__global const uint* a, uint n, __global uint* result) {
    size_t i = get_global_id(0);
    size_t li = get_local_id(0);
    size_t groupSize = get_local_size(0);
    size_t offset = (i - li) * WPT;

    if (offset >= n)
        return;

    uint sum = 0;
    for (size_t j = 0; j < WPT && offset + j < n; ++j)
        sum += a[offset + j * groupSize + li];

    atomic_add(result, sum);
}

__kernel void sumMajorWorker(__global const uint* a, uint n, __global uint* result) {
    size_t i = get_global_id(0);
    size_t li = get_local_id(0);
    size_t groupSize = get_local_size(0);

    __local uint buf[256];
    if (i < n)
        buf[li] = a[i];
    else
        buf[li] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    uint sum = 0;
    if (li == 0) {
        uint end = min(groupSize, n - i);
        for (size_t j = 0; j < end; ++j)
            sum += buf[j];
        atomic_add(result, sum);
    }
}

void sumLocalTree(__global const uint* a, uint n, __local uint* buf) {
    size_t i = get_global_id(0);
    size_t li = get_local_id(0);
    size_t groupSize = get_local_size(0);

    if (i < n)
        buf[li] = a[i];
    else
        buf[li] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t w = groupSize; w > 1; w /= 2) {
        if (li * 2 < w) {
            buf[li] += buf[li + w / 2];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void sumTree(__global const uint* a, uint n, __global uint* result) {
    size_t li = get_local_id(0);

    __local uint buf[256];
    sumLocalTree(a, n, buf);

    if (li == 0)
        atomic_add(result, buf[0]);
}

__kernel void sumTreeRecursive(__global const uint* a, uint n, __global uint* result) {
    size_t li = get_local_id(0);
    size_t gi = get_group_id(0);

    __local uint buf[256];
    sumLocalTree(a, n, buf);

    if (li == 0)
        result[gi] = buf[0];
}
