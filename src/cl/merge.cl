#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define ARRAY_SIZE 32 * 1024 * 1024

#define BIN_SIZE 1024

#define BIN_COUNT 32

__kernel void merge(__global const float* as,
                    __global float* bs,
                    unsigned int n,
                    unsigned int level_size)
{
    int index = get_global_id(0);
    int part_index = index / level_size;
    int local_index = index % level_size;

    if (index >= n)
        return;

    int part_begin = part_index * level_size;
    int part_center = part_begin + level_size / 2;
    int part_end = min(part_begin + level_size, n);
    int left_size = part_center - part_begin;
    int right_size = part_end - part_center;
    if (right_size == 0) {
        bs[index] = as[index];
        return;
    }

    int l = max(-1, local_index - right_size - 1), r = min(local_index, left_size);
    while (r - l > 1) {
        int m = (l + r) / 2;
        int left_index = m;
        int right_index = local_index - m - 1;
        if (as[part_begin + left_index] < as[part_center + right_index]) {
            l = m;
        } else {
            r = m;
        }
    }

    if (r == local_index - right_size || (r != left_size && as[part_begin + r] < as[part_center + local_index - r])) {
        bs[index] = as[part_begin + r];
    } else {
        bs[index] = as[part_center + local_index - r];
    }
}
