#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void merge(__global const float *a, __global float *result, unsigned int m, unsigned int n) {
    unsigned int id = get_global_id(0);
    if (id >= n) {
        return;
    }

    int shift = id / (2 * m) * (2 * m);
    id -= shift;
    if (id == 2 * m - 1) {
        result[shift] = a[shift] < a[shift + m] ? a[shift] : a[shift + m];
        return;
    }

    int l = 0;
    int index_shift = id < m ? 0 : id - m + 1;
    int r = id < m? id : id - 2*index_shift;
    int index_i = l + index_shift;
    int index_j = id - l - index_shift;

    if (a[index_i + shift] < a[index_j + shift + m]) {
        index_i = r + index_shift;
        index_j = id - r - index_shift;
        if (a[index_i + shift] >= a[index_j + shift + m]) {
            while (r - l > 1) {
                int mid = (r + l) / 2;
                index_i = mid + index_shift;
                index_j = id - mid - index_shift;

                if (a[index_i + shift] >= a[index_j + shift + m]) {
                    r = mid;
                } else {
                    l = mid;
                }
            }
            index_i = r + index_shift;
            index_j = id - r - index_shift;
        } else {
            index_i++;
            index_j--;
        }
    }

    float res;
    if (index_i == m) {
      res = a[index_j + shift + m + 1];
    } else if (index_j == m - 1) {
      res = a[index_i + shift];
    } else if (a[index_i + shift] < a[ index_j + shift + m + 1]) {
      res = a[index_i + shift];
    } else {
      res = a[index_j + shift + m + 1];
    }

    result[shift + id + 1] = res;
}