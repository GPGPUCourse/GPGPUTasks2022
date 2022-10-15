

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
    size_t r = l + mergeSize;
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
