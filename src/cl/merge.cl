#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define ARRAY_SIZE 32 * 1024 * 1024

__kernel void merge(__global float* as,
                    __global float* b,
                    unsigned int n)
{
    int index = get_global_id(0);

//    as[index] = 0;

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (index >= n) {
        return;
    }

//    __global float b[ARRAY_SIZE];

    for (int level_size = 2; level_size < n*2; level_size*=2) {

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (index >= n / level_size) {
            return;
        }

        int part_begin = index * level_size;
        int part_center = part_begin + level_size / 2;
        int part_end = part_begin + level_size;

//        as[part_begin] = index;

//        if (part_begin < n)
//            as[index] = part_end;
//        as[part_begin] = index;
//        for (int i = part_begin; i < part_end && i < n; i++) {
//            as[i] = index; //b[i];
//        }

        if (part_end > n) {
            part_end = n;
        }

        if (part_begin >= n)
            continue;
        if (part_center >= n)
            continue;


//        barrier(CLK_GLOBAL_MEM_FENCE);

//        int bin_per_part = 1; //get_global_size(0) / (1<<level);
//
//        int bin_borders_1[2];
//        int bin_borders_2[2];
//
//        int bin_borders_sum[2];
//
//        bin_borders_1[0] =            part_begin;
//        bin_borders_1[bin_per_part] = part_center;
//        bin_borders_2[0] =            part_center;
//        bin_borders_2[bin_per_part] = part_end;

//        for (int bin = 0; bin < bin_per_part-1; bin++) {
//
////                int diag = 4;
////
////                int l = 0, r = diag+1;
////                while (l <= r) {
////                    int m = (l + r) / 2;
////                    if(as[part_begin + m] < as[part_center + diag - m]) {
////                        l = m;
////                    } else {
////                        r = m+1;
////                    }
////                }
////                bin_borders_1[bin+1] = part_begin + l;
////                bin_borders_2[bin+1] = part_center + diag - l;
//
//
//        }

//        bin_borders_sum[0] = part_begin;
//        for (int bin = 1; bin < bin_per_part-1; bin++) {
//            bin_borders_sum[bin] = bin_borders_sum[bin-1] + (bin_borders_1[bin] - bin_borders_1[bin-1]) + (bin_borders_2[bin] - bin_borders_2[bin-1]);
//        }

        barrier(CLK_GLOBAL_MEM_FENCE);

//        for (int bin = 0; bin < bin_per_part; bin++) {

        int bin_start_1 = part_begin; //bin_borders_1[bin];
        int bin_end_1 = part_center; //bin_borders_1[bin+1];
        int bin_start_2 = part_center; //bin_borders_2[bin];
        int bin_end_2 = part_end; //bin_borders_2[bin+1];

        int i = part_begin, j = part_center;
        int k = part_begin;
        while (i < part_center && j < part_end) {
            if (as[i] < as[j])
                b[k++] = as[i++];
            else
                b[k++] = as[j++];
        }
        while (i < part_center)
            b[k++] = as[i++];
        while (j < part_end)
            b[k++] = as[j++];


        barrier(CLK_GLOBAL_MEM_FENCE);

        for (int i = part_begin; i < part_end && i < n; i++) {
            as[i] = b[i];
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

    }
}
