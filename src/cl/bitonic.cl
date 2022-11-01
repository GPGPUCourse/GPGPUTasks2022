
#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void bitonic(__global float *as, unsigned int N, unsigned int chunk_size, unsigned int global_chunk_size,
                      int use_local_memory /*, __local float *storage */ )
{

    __local float storage[1024];

    unsigned int i = get_global_id(0);
    unsigned int hcs = chunk_size / 2;

    unsigned int group_size = get_local_size(0);

    bool ascending = ((i / (global_chunk_size / 2)) % 2) == 0;

    if (use_local_memory) {
        unsigned int offset = i / group_size * group_size * 2;
        unsigned int local_id = get_local_id(0);
        unsigned int global_id = offset + i % group_size;
        storage[local_id] = as[global_id];
        storage[local_id + group_size] = as[global_id + group_size];
        barrier(CLK_LOCAL_MEM_FENCE);

        while (chunk_size > 1) {
            unsigned int ind1 = i / (chunk_size / 2) * chunk_size + i % (chunk_size / 2) - offset;
            unsigned int ind2 = ind1 + chunk_size / 2;
            float elem1 = storage[ind1];
            float elem2 = storage[ind2];
            if ((ascending && elem1 > elem2) || (!ascending && elem1 < elem2)) {
                storage[ind1] = elem2;
                storage[ind2] = elem1;
            }

            chunk_size /= 2;

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        as[global_id] = storage[local_id];
        as[global_id + group_size] = storage[local_id + group_size];
    } else {
        unsigned int global_id = i / hcs * chunk_size + i % hcs;
        float elem1 = as[global_id];
        float elem2 = as[global_id + hcs];
        if ((ascending && elem1 > elem2) || (!ascending && elem1 < elem2)) {
            as[global_id] = elem2;
            as[global_id + hcs] = elem1;
        }
    }

}