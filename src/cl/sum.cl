#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALUES_PER_WORK_ITEM 64
#define WORK_GROUP_SIZE 128
__kernel void atomic_sum(__global unsigned int* res, __global const unsigned int* data, const unsigned int n) {
   const unsigned int index = get_global_id(0);
   if (index < n) {
      atomic_add(res, data[index]);
   }
}

__kernel void iter_sum(__global unsigned int* res, __global const unsigned int* data, const unsigned int n) {
   const unsigned int index = get_global_id(0);
   unsigned int sum = 0;
   for (int i = 0; i < VALUES_PER_WORK_ITEM; i++) {
      sum += data[index * VALUES_PER_WORK_ITEM + i];
   }
   atomic_add(res, sum);
}

__kernel void coalesce_sum(__global unsigned int* res, __global const unsigned int* data, const unsigned int n) {
   const unsigned int local_id = get_local_id(0);
   const unsigned int group_id = get_group_id(0);
   const unsigned int group_size = get_local_size(0);

   unsigned int sum = 0;
   for (int i = 0; i < VALUES_PER_WORK_ITEM; i++) {
      sum += data[group_id * group_size * VALUES_PER_WORK_ITEM + i * group_size + local_id];
   }
   atomic_add(res, sum);
}

__kernel void local_sum(__global unsigned int* res, __global const unsigned int* data, const unsigned int n) {
   const unsigned int local_id = get_local_id(0);
   const unsigned int global_id = get_global_id(0);

   __local unsigned int local_data[WORK_GROUP_SIZE];
   local_data[local_id] = data[global_id];
   barrier(CLK_LOCAL_MEM_FENCE);
   if (local_id == 0) {
      unsigned int sum = 0;
      for (int i = 0; i < WORK_GROUP_SIZE; i++) {
         sum += local_data[i];
      }
      atomic_add(res, sum);
   }
}

__kernel void tree_sum(__global unsigned int* res, __global const unsigned int* data, const unsigned int n) {
   const unsigned int local_id = get_local_id(0);
   const unsigned int global_id = get_global_id(0);

   __local unsigned int local_data[WORK_GROUP_SIZE];
   local_data[local_id] = data[global_id];
   barrier(CLK_LOCAL_MEM_FENCE);
   for (int nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
      if (2 * local_id < nvalues) {
         int a = local_data[local_id];
         int b = local_data[local_id + nvalues/2];
         local_data[local_id] = a + b;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }
   if (local_id == 0) {
      atomic_add(res, local_data[0]);
   }
}
