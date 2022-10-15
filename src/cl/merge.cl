__kernel void mergeLocal(__global float* in, uint size) {
    size_t id = get_global_id(0);
    size_t lId = get_local_id(0);
    size_t lSize = min((size_t)get_local_size(0), size - (id - lId));

    __local float bufA[256], bufB[256];
    __local float* lIn = bufA;
    __local float* lOut = bufB;
    if (id < size) {
        lIn[lId] = in[id];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t mergeSize = 1; mergeSize < lSize; mergeSize *= 2) {
        if (lId < lSize) {
            size_t mPartId = lId / mergeSize;
            size_t mCounterpartId = mPartId ^ 1;
            size_t mPartOffset = lId % mergeSize;
            size_t mPartStart = lId - mPartOffset;
            size_t mCounterpartStart = mCounterpartId * mergeSize;

            float x = lIn[lId];

            size_t l = 0;
            size_t r = mergeSize;
            while (l < r) {
                size_t m = (l + r) / 2;

                if (mCounterpartStart + m < lSize) {
                    float y = lIn[mCounterpartStart + m];
                    if (x > y || x == y && (mPartId & 1)) {
                        l = m + 1;
                        continue;
                    }
                }

                r = m;
            }

            lOut[min(mPartStart, mCounterpartStart) + mPartOffset + l] = x;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        __local float* t = lIn;
        lIn = lOut;
        lOut = t;
    }

    if (id < size) {
        in[id] = lIn[lId];
    }
}

__kernel void merge(__global float* in, uint mergeSize, uint size, __global float* out) {
    size_t id = get_global_id(0);

    if (id >= size) {
        return;
    }

    size_t mPartId = id / mergeSize;
    size_t mCounterpartId = mPartId ^ 1;
    size_t mPartOffset = id % mergeSize;
    size_t mPartStart = id - mPartOffset;
    size_t mCounterpartStart = mCounterpartId * mergeSize;

    float x = in[id];

    size_t l = 0;
    size_t r = mergeSize;
    while (l < r) {
        size_t m = (l + r) / 2;

        if (mCounterpartStart + m < size) {
            float y = in[mCounterpartStart + m];
            if (x > y || x == y && (mPartId & 1)) {
                l = m + 1;
                continue;
            }
        }

        r = m;
    }

    out[min(mPartStart, mCounterpartStart) + mPartOffset + l] = x;
}
