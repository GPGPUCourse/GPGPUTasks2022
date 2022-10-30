#define WORK_GROUP_SZ 256
#define MAX_LOCAL_BLOCK_ORDER 8

__kernel void bitonic_local(__global float *as, uint n) {
    size_t id = get_global_id(0);
    size_t lId = get_local_id(0);

    __local float buf[WORK_GROUP_SZ * 2];
    if (id < n) {
        buf[lId] = as[(id - lId) * 2 + lId];
    }
    if (id + WORK_GROUP_SZ < n) {
        buf[lId + WORK_GROUP_SZ] = as[(id - lId) * 2 + lId + WORK_GROUP_SZ];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t blockOrder = 0; blockOrder <= MAX_LOCAL_BLOCK_ORDER; ++blockOrder) {
        for (size_t stepOrder = blockOrder + 1; stepOrder >= 1; --stepOrder) {
            size_t comparisonBlockOrder = stepOrder - 1;
            bool compareGreater = id & (1 << blockOrder);
            size_t comparisonBlockMask = (((size_t) 1) << comparisonBlockOrder) - 1;
            size_t compareBlockStart = ((lId & ~comparisonBlockMask) << 1) | (lId & comparisonBlockMask);
            size_t i1 = compareBlockStart;
            size_t i2 = compareBlockStart | (1 << comparisonBlockOrder);
            if (i2 + (id - lId) * 2 < n) {
                bool cmp = buf[i1] < buf[i2];
                if ((compareGreater && cmp) || (!compareGreater && !cmp)) {
                    float t = buf[i1];
                    buf[i1] = buf[i2];
                    buf[i2] = t;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (id < n) {
        as[(id - lId) * 2 + lId] = buf[lId];
    }
    if (id + WORK_GROUP_SZ < n) {
        as[(id - lId) * 2 + lId + WORK_GROUP_SZ] = buf[lId + WORK_GROUP_SZ];
    }
}

__kernel void bitonic(__global float *as, uint n, uint blockOrder, uint comparisonBlockOrder) {
    size_t id = get_global_id(0);
    bool compareGreater = id & (1 << blockOrder);
    size_t comparisonBlockMask = (((size_t)1) << comparisonBlockOrder) - 1;
    size_t compareBlockStart = ((id & ~comparisonBlockMask) << 1) | (id & comparisonBlockMask);
    size_t i1 = compareBlockStart;
    size_t i2 = compareBlockStart | (1 << comparisonBlockOrder);
    if (i2 < n) {
        bool cmp = as[i1] < as[i2];
        if ((compareGreater && cmp) || (!compareGreater && !cmp)) {
            float t = as[i1];
            as[i1] = as[i2];
            as[i2] = t;
        }
    }
}
