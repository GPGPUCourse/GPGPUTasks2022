#define TILE_SIZE 16
__kernel void matrix_transpose(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0); // column id
    int j = get_global_id(1); // row id
    __local float tile[TILE_SIZE][TILE_SIZE + 1]; // one element in every row for fake element which fix bank-conflicts
    int local_i = get_local_id(0); // local column id
    int local_j = get_local_id(1); // local row id

    // 1. copy from global to local
    float element;
    if (j < m && i < k) {
        element = a[j * k + i]; // coalesced reading because in warp j is const and only is changing
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. transpose locally
    if (j < m && i < k) {
        tile[local_i][local_j] = element; // no bank conflicts because we add fake element to every row and that is
                                          // why access to elements from one column -- local_j is const in warp
                                          // -- is not access to one bank (every [local_i][local_j] matches own bank
                                          // for every thread in one warp because of fake element in every row of tile)
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3. save result in global
    int transposing_element_old_global_i = i - local_i + local_j;
    int transposing_element_old_global_j = j - local_j + local_i;
    if (transposing_element_old_global_j < m && transposing_element_old_global_i < k) {
        // no bank conflicts because local_j is const in one warp and local_i matches own bank for every thread
        // in one warp
        // coalesced writing because old_i is const in warp (because i - local_i is const and local_j const in warp)
        // and old_j matches own bank for every thread in one warp
        at[transposing_element_old_global_i * m + transposing_element_old_global_j] = tile[local_j][local_i];
    }
}