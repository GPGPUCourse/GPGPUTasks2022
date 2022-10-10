__kernel void matrix_transpose(__global float* a, __global float* at, uint m, uint k) {
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

    size_t li = get_local_id(0);
    size_t lj = get_local_id(1);

    __local float buf[17][17];
    if (i < k && j < m)
        buf[li][lj] = a[j * k + i]; // [i][j]

    barrier(CLK_LOCAL_MEM_FENCE);

    size_t iOffset = i - li;
    size_t jOffset = j - lj;

    if (iOffset + lj < k && jOffset + li < m)
        at[(iOffset + lj) * m + jOffset + li] = buf[lj][li]; // [j][i]
}
