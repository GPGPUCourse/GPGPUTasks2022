#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALUES_PER_WORK_ITEM 64
#define WORK_GROUP_SIZE 128
__kernel void sum_gpu_1(__global const unsigned int* xs, int n, __global unsigned int* res) {
   int idx = get_global_id(0);
   if (idx >= n)
      return;

   atomic_add(res, xs[idx]);
}

__kernel void sum_gpu_2(__global const unsigned int* xs, int n, __global unsigned int* res) {
   int id = get_global_id(0);
   unsigned int sum = 0;
   for (size_t i = 0; i < VALUES_PER_WORK_ITEM; i++) {
      int idx = id * VALUES_PER_WORK_ITEM + i;
      if (idx < n) {
         sum += xs[idx];
      }
   }
   atomic_add(res, sum);
}

__kernel void sum_gpu_3(__global const unsigned int* xs, int n, __global unsigned int* res) {
   int local_id = get_local_id(0);
   int group_id = get_group_id(0);
   int group_size = get_local_size(0);

   unsigned int sum = 0;
   for (int i = 0; i < VALUES_PER_WORK_ITEM; i++) {
      int idx = group_id * group_size * VALUES_PER_WORK_ITEM + i * group_size + local_id;
      if (idx < n) {
         sum += xs[group_id * group_size * VALUES_PER_WORK_ITEM + i * group_size + local_id];
      }
   }

   atomic_add(res, sum);
}

__kernel void sum_gpu_4(__global const unsigned int* xs, int n, __global unsigned int* res) {
   int local_id = get_local_id(0);
   int global_id = get_global_id(0);

   __local unsigned int local_xs[WORK_GROUP_SIZE];
   if (global_id < n) {
       local_xs[local_id] = xs[global_id];
   } else {
       local_xs[local_id] = 0;
   }

   barrier(CLK_LOCAL_MEM_FENCE);
   if (local_id == 0) {
      unsigned int sum = 0;
      for (size_t i = 0; i < WORK_GROUP_SIZE; i++) {
         sum += local_xs[i];
      }
      atomic_add(res, sum);
   }
}

__kernel void sum_gpu_5(__global const unsigned int* xs, int n, __global unsigned int* res) {
   int local_id = get_local_id(0);
   int global_id = get_global_id(0);

   __local unsigned int local_xs[WORK_GROUP_SIZE];
    if (global_id < n) {
        local_xs[local_id] = xs[global_id];
    } else {
        local_xs[local_id] = 0;
    }

   barrier(CLK_LOCAL_MEM_FENCE);
   for (int nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
      if (2 * local_id < nvalues) {
         int a = local_xs[local_id];
         int b = local_xs[local_id + nvalues/2];
         local_xs[local_id] = a + b;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }
   if (local_id == 0) {
      atomic_add(res, local_xs[0]);
   }
}