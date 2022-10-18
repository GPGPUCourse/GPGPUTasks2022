#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

__kernel void merge(__global const float *as, __global float *bs, const unsigned int size, const unsigned int k) {
    unsigned i = get_global_id(0);
    if (i >= size) {
        return;
    }

    if (i >= size) {
        return;
    }

    unsigned int al = i / (k * 2) * k * 2;
    unsigned int ar = al + k;
    unsigned int bl = ar;
    unsigned int br = bl + k;

    unsigned int ind = i - al;
    int b = bl + ind;

    int left = max((int) ind - (int) k, 0);
    int right = min(k, ind);

    while (left < right) {
        int mid = (left + right) / 2;
        if (as[al + mid] <= as[b - mid - 1]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    bs[i] = (al + right < ar && (b - right >= br || as[al + right] <= as[b - right])) ? as[al + right] : as[b - right];
}
