#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define ARRAY_SIZE 32 * 1024 * 1024

#define BIN_SIZE 1024

#define BIN_COUNT 32

__kernel void merge(__global float* as,
                    __global float* bs,
                    unsigned int n,
                    unsigned int level_size)
{
    int parts_count = n / level_size;

    int part_index = get_global_id(0);

    if (part_index >= parts_count)
        return;

    barrier(CLK_GLOBAL_MEM_FENCE);

    int part_begin = part_index * level_size;
    int part_center = part_begin + level_size / 2;
    int part_end = part_begin + level_size;

    barrier(CLK_GLOBAL_MEM_FENCE); // LOCAL?

    int i = part_begin, j = part_center;
    int k = part_begin;
    while (i < part_center && j < part_end && k < part_end) {
        if (as[i] < as[j])
            bs[k++] = as[i++];
        else
            bs[k++] = as[j++];
    }
    while (i < part_center && k < part_end)
        bs[k++] = as[i++];
    while (j < part_end && k < part_end)
        bs[k++] = as[j++];

    for (int i = part_begin; i < part_end && i < n; i++) {
        as[i] = bs[i];
    }


}
