#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

int find_merge_path(
    __global const float *first,
    __global const float *second,
    unsigned int n,
    unsigned int m,
    int position
) {
    int l = -1;
    int r = position + 1;

    while (l + 1 < r) {
        int mid = (l + r) / 2;
        int i0 = position - mid;
        int i1 = mid;
        if (i0 >= n || (i1 < m && first[i0] >= second[i1])) {
            l = mid;
        } else {
            r = mid;
        }
    }


    int i0 = position - l;
    int i1 = l;

    if (i1 == -1) {
        return position;
    }
    if (i0 == 0) {
        return n + position;
    }

    if (first[i0 - 1] >= second[i1]) {
        return i0 - 1;
    } else {
        return n + i1;
    }

}

__kernel void merge(
    __global const float *array,
    __global float *result,
    unsigned int n,
    unsigned int part_size
) {
    const int idx = get_global_id(0);
    if (idx >= n) {
        return;
    }

    const int level_size = part_size * 2;
    const int first_start = idx / level_size * level_size;
    const int second_start = first_start + part_size;
    const int level_idx = idx % level_size;

    int first_size = part_size;
    if (first_start + first_size >= n) {
        first_size = n - first_start;
    }

    int second_size = part_size;
    if (second_start + second_size >= n) {
        second_size = n - second_start;
    }

    if (second_size == 0 || first_size == 0) {
        result[idx] = array[idx];
        return;
    }

    int level_from_idx = find_merge_path(array + first_start, array + second_start, first_size, second_size, level_idx);

    result[idx] = array[first_start + level_from_idx];
}
