#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define ARRAY_SIZE 32 * 1024 * 1024

#define BIN_SIZE 1024

#define BIN_COUNT 32

__kernel void merge(__global float* as,
                    unsigned int n,
                    unsigned int level_size)
{
    int parts_count = n / level_size;
    int near_bin_size = BIN_SIZE / 4;
    int bin_per_part = 1; level_size/2 / near_bin_size; //get_global_size(0) / (1<<level);
    if (bin_per_part == 0)
        bin_per_part = 1;
    if (bin_per_part > BIN_COUNT)
        bin_per_part = BIN_COUNT;
    if (bin_per_part > get_local_size(0))
        bin_per_part = get_local_size(0);

    int part_index = get_group_id(0);
    int bin_index = get_local_id(0);


    if (bin_index >= bin_per_part)
        return;
    if (part_index >= parts_count)
        return;

    barrier(CLK_GLOBAL_MEM_FENCE);

    int part_begin = part_index * level_size;
    int part_center = part_begin + level_size / 2;
    int part_end = part_begin + level_size;

//        barrier(CLK_GLOBAL_MEM_FENCE);

    barrier(CLK_LOCAL_MEM_FENCE);

    __local int bin_borders_1[BIN_COUNT+1];
    __local int bin_borders_2[BIN_COUNT+1];
    __local int bin_borders_sum[BIN_COUNT+1];

    if (bin_index == 0) {
        bin_borders_1[0] =            part_begin;
        bin_borders_1[bin_per_part] = part_center;
        bin_borders_2[0] =            part_center;
        bin_borders_2[bin_per_part] = part_end;

        for (int bin = 0; bin < bin_per_part - 1; bin++) {

            int diag = near_bin_size * 2 * (bin + 1);

            int l = 0, r = diag;
            while (l < r) {
                int m = (l + r) / 2;
                if (as[part_begin + m] <= as[part_center + diag - m]) {
                    l = m + 1;
                } else {
                    r = m;
                }
            }
            bin_borders_1[bin + 1] = part_begin + l;
            bin_borders_2[bin + 1] = part_center + diag - l;

        }

        bin_borders_sum[0] = part_begin;
        for (int bin = 1; bin < bin_per_part+1; bin++) {
            if (bin_borders_sum[bin] - bin_borders_sum[bin-1] > BIN_SIZE)
                return;
            bin_borders_sum[bin] = bin_borders_sum[bin-1] + (bin_borders_1[bin] - bin_borders_1[bin-1]) + (bin_borders_2[bin] - bin_borders_2[bin-1]);
        }

    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float bin_1[BIN_SIZE];
    float bin_2[BIN_SIZE];
    float bin_sorted[BIN_SIZE];


    int bin_start_1 = bin_borders_1[bin_index];
    int bin_end_1 = bin_borders_1[bin_index + 1];
    int bin_start_2 = bin_borders_2[bin_index];
    int bin_end_2 = bin_borders_2[bin_index + 1];

    for (int i = bin_start_1; i < bin_end_1; i++)
        bin_1[i - bin_start_1] = as[i];
    for (int i = bin_start_2; i < bin_end_2; i++)
        bin_2[i - bin_start_2] = as[i];

    barrier(CLK_GLOBAL_MEM_FENCE); // LOCAL?

    int i = 0, j = 0;
    int k = 0;
    while (i < bin_end_1 - bin_start_1 && j < bin_end_2 - bin_start_2 && k < BIN_SIZE) {
        if (bin_1[i] < bin_2[j])
            bin_sorted[k++] = bin_1[i++];
        else
            bin_sorted[k++] = bin_2[j++];
    }
    while (i < bin_end_1 - bin_start_1 && k < BIN_SIZE)
        bin_sorted[k++] = bin_1[i++];
    while (j < bin_end_2 - bin_start_2 && k < BIN_SIZE)
        bin_sorted[k++] = bin_2[j++];

    for (int i = bin_borders_sum[bin_index]; i < bin_borders_sum[bin_index+1] && i < n; i++) {
//                as[i] = bin_start_1; bin_tmp[i - bin_borders_sum[bin]];
        as[i] = bin_sorted[i - bin_borders_sum[bin_index]];
    }


}
