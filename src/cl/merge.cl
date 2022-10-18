#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void merge(__global float *as, __global float *as_merged,
                    unsigned int merge_part_size, unsigned int n) {
    const unsigned int global_i = get_global_id(0);
    if (global_i >= n)
        return;

    unsigned int offset = global_i / (merge_part_size * 2) * merge_part_size * 2;
    unsigned int diag_index = global_i - offset;

    __global float *fst_part = as + offset;
    unsigned int fst_part_size = merge_part_size;
    __global float *snd_part = as + offset + fst_part_size;
    unsigned int snd_part_size = min(global_i + fst_part_size + merge_part_size, n) - global_i - fst_part_size;

    unsigned int left = max(0, (int)diag_index - (int)merge_part_size);
    unsigned int right = min(fst_part_size, diag_index);

    while (left < right) {
        int med = (left + right) / 2;
        if (fst_part[med] > snd_part[diag_index - med - 1]) {
            right = med;
        } else {
            left =  med + 1;
        }
    }

    if ((fst_part_size + right <= diag_index || fst_part[right] <= snd_part[diag_index - right]) && right != fst_part_size) {
        as_merged[global_i] = fst_part[right];
    } else {
        as_merged[global_i] = snd_part[diag_index - right];
    }
}