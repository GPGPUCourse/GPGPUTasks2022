typedef struct Mtx {
    __global float* a;
    size_t m;
    size_t n;
} Mtx;

inline float mtx_get(Mtx m, uint i, uint j) {
    return m.a[j * m.n + i];
}

inline void mtx_set(Mtx m, uint i, uint j, float v) {
    m.a[j * m.n + i] = v;
}

inline Mtx mtx_create(__global float* a, size_t m, size_t n) {
    Mtx res;
    res.a = a;
    res.m = m;
    res.n = n;
    return res;
}


__kernel void matrix_multiplication(__global float* A, __global float* B, __global float* C,
    uint M, uint K, uint N) {
    size_t i = get_global_id(0); // 0..N
    size_t j = get_global_id(1); // 0..M

    if (i >= N || j >= M)
        return;

    size_t li = get_local_id(0);
    size_t lj = get_local_id(1);

    size_t tileSize = get_local_size(0);
    size_t numTiles = K / tileSize;

    Mtx a = mtx_create(A, M, K);
    Mtx b = mtx_create(B, K, N);
    Mtx c = mtx_create(C, M, N);

    __local float tileA[17][17];
    __local float tileB[17][17];

    float sum = 0.0f;

    for (size_t t = 0; t < numTiles; ++t) {
        tileA[lj][li] = mtx_get(a, t * tileSize + li, j);
        tileB[li][lj] = mtx_get(b, i, t * tileSize + lj);

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t l = 0; l < tileSize; ++l) {
            sum += tileA[lj][l] * tileB[li][l];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    mtx_set(c, i, j, sum);
}


__kernel void matrix_multiplication2(__global float* A, __global float* B, __global float* C,
    uint M, uint K, uint N) {
    size_t i = get_global_id(0); // 0..N
    size_t j = get_global_id(1); // 0..M/stripeSize

    if (i >= N || j >= M)
        return;

    size_t li = get_local_id(0); // 0..tileSize
    size_t lj = get_local_id(1); // 0..tileSize/stripeSize

    size_t tileSize = get_local_size(0);
    size_t numTileStripes = get_local_size(1);
    size_t stripeSize = tileSize / numTileStripes;
    size_t numTiles = K / tileSize;

    Mtx a = mtx_create(A, M, K);
    Mtx b = mtx_create(B, K, N);
    Mtx c = mtx_create(C, M, N);

    __local float tileA[33][33];
    __local float tileB[33][33];
    float sum[16];
    for (size_t w = 0; w < stripeSize; ++w) {
        sum[w] = 0.0f;
    }

    for (size_t t = 0; t < numTiles; ++t) {
        for (size_t w = 0; w < stripeSize; ++w) {
            tileA[lj * stripeSize + w][li] = mtx_get(a, t * tileSize + li, j * stripeSize + w);
            tileB[li][lj * stripeSize + w] = mtx_get(b, i, t * tileSize + lj * stripeSize + w);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t l = 0; l < tileSize; ++l) {
            float tmp = tileB[li][l];
            for (size_t w = 0; w < stripeSize; ++w) {
                sum[w] += tileA[lj * stripeSize + w][l] * tmp;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (size_t w = 0; w < stripeSize; ++w) {
        mtx_set(c, i, j * stripeSize + w, sum[w]);
    }
}
