#define TILE_SIZE 16
__kernel void matrix_transpose(__global const float *as, __global float *as_t, unsigned int M, unsigned int K)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    tile[local_j][local_i] = as[j * K + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    int x = i - local_i + local_j;
    int y = j - local_j + local_i;

    as_t[x * M + y] = tile[local_i][local_j];
}