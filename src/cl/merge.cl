#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void merge(__global const float* a, __global float* b, const unsigned int n, const unsigned int group_size) {
   const unsigned int global_id = get_global_id(0);

   if (global_id >= n) return;

   const unsigned int offset = global_id / (2 * group_size) * 2 * group_size;
   const unsigned int offset_index = global_id % (2 * group_size);

   const unsigned int a_i = offset;
   const unsigned int a_j = min(n, offset + group_size);

   if (a_j >= n) {
      b[global_id] = a[global_id];
      return;
   }

   const unsigned int b_i = a_j;
   const unsigned int b_j = min(n, b_i + group_size);
   int l = - 1;
   int r = group_size;
   while (r - l > 1) {
      const unsigned int m = (l + r) / 2;
      if (offset_index < group_size) {
         if (a[global_id] < a[offset + group_size + m]) {
            r = m;
         } else {
            l = m;
         }
      } else {
         if (a[global_id] > a[offset + m]) {
            l = m;
         } else {
            r = m;
         }
      }
   }
   b[offset + (offset_index % group_size) + (l + 1)] = a[global_id];
}