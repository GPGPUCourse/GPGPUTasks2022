#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6


__kernel void merge(__global float *as, __global float *bs, int k) {
    int i = get_global_id(0);
    int offset = i / (k * 2) * (k * 2);
    int local_i = i - offset;
    int l = max(local_i - k, 0);
    int r = min(local_i, k);
    while (l < r) {
        int m = (l + r) / 2;
        if (as[offset + m] <= as[i + k - m - 1])
            l = m + 1;
        else
            r = m;
    }
    bs[i] = (k + r <= local_i || as[offset + r] <= as[i + k - r]) && (r != k) ? as[offset + r] : as[i + k - r];
}
