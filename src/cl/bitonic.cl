__kernel void bitonic(__global float *as, uint n, uint blockOrder, uint comparisonStep) {
    size_t id = get_global_id(0);
    bool compareGreater = id & (1 << blockOrder);
    size_t compareBlockId = id / comparisonStep;
    size_t idInCompareBlock = id % comparisonStep;
    size_t compareBlockStart = compareBlockId * comparisonStep * 2 + idInCompareBlock;
    size_t i1 = compareBlockStart;
    size_t i2 = compareBlockStart + comparisonStep;
    if (i2 < n) {
        bool cmp = as[i1] < as[i2];
        if ((compareGreater && cmp) || (!compareGreater && !cmp)) {
            float t = as[i1];
            as[i1] = as[i2];
            as[i2] = t;
        }
    }
}
