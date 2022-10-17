
#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6


__kernel void merge(__global float *as, __global float *bs, unsigned int N, unsigned int lvl_size)
{

    unsigned int i = get_global_id(0);

    if (lvl_size == 2) {
        if (i < N / 2) {
            if (as[i * 2] < as[i * 2 + 1]) {
                bs[i * 2] = as[i * 2];
                bs[i * 2 + 1] = as[i * 2 + 1];
            } else {
                bs[i * 2] = as[i * 2 + 1];
                bs[i * 2 + 1] = as[i * 2];
            }
        }
    } else {
        unsigned int hls = lvl_size / 2;
        unsigned int start_ind;
        float elem = as[i];
        bool move_eq = false;
        if (((i / hls) % 2) == 0) {
            start_ind = i - i % hls + hls;
            move_eq = true;
        } else {
            start_ind = i - i % hls - hls;
        }

        unsigned int l = -1;
        unsigned int r = hls;
        while (r - l != 1) {
            unsigned int m = (r + l) / 2;

            float val = as[start_ind + m];
            if (val < elem || (val == elem && move_eq)) {
                l = m;
            } else {
                r = m;
            }
        }

        unsigned int pos = r + (i % hls);
        bs[i - i % lvl_size + pos] = elem;
    }
}