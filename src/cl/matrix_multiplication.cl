#define CHUNK_SIZE 16
#define WORK_PER_THREAD 4

__kernel void matrix_multiplication_chunked(__global const int* A,
                                            __global const int* B,
                                            __global       int* C,
                                            unsigned int M, unsigned int K, unsigned int N)
{
  __local int A_local[CHUNK_SIZE][CHUNK_SIZE];
  __local int B_local[CHUNK_SIZE][CHUNK_SIZE];

  int j_local = get_local_id(0);
  int i_local = get_local_id(1);
  int j = get_group_id(0) * CHUNK_SIZE + j_local;
  int i = get_group_id(1) * CHUNK_SIZE + i_local;

  int result = 0;
  for (int k_base = 0; k_base < K; k_base += CHUNK_SIZE) {
    int a_j = k_base + j_local;
    int b_i = k_base + i_local;
    A_local[i_local][j_local] = (i < M && a_j < K) ? A[i * K + a_j] : 0;
    B_local[i_local][j_local] = (j < N && b_i < K) ? B[b_i * N + j] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k_local = 0; k_local < CHUNK_SIZE; ++k_local) {
      result += A_local[i_local][k_local] * B_local[k_local][j_local];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (i < M && j < N) {
    C[i * N + j] = result;
  }
}

/*
              ....................----....
              ....................----....
              ....................----....
              ....................----....
           K  ....................==##....
              ....................==##....
              ....................==##....
              ....................==##....
              ....................----....
     K                     N
 .........    ............................
 .........    ............................
 ......... M  ............................
 .........    ............................
 ----====-    ....................====....
 ----####-    ....................==##....
 */

__kernel void matrix_multiplication_multijob(__global const int* A,
                                             __global const int* B,
                                             __global       int* C,
                                             unsigned int M, unsigned int K, unsigned int N)
{
  __local int A_local[CHUNK_SIZE][CHUNK_SIZE];
  __local int B_local[CHUNK_SIZE][CHUNK_SIZE];

  int j_local_block = get_local_id(0) * WORK_PER_THREAD;
  int i_local = get_local_id(1);
  int j_block = get_group_id(0) * CHUNK_SIZE + j_local_block;
  int i = get_group_id(1) * CHUNK_SIZE + i_local;

  int result[WORK_PER_THREAD];
  for (int j_wpt = 0; j_wpt < WORK_PER_THREAD; ++j_wpt) {
    result[j_wpt] = 0;
  }

  for (int k_base = 0; k_base < K; k_base += CHUNK_SIZE) {
    for (int j_wpt = 0, j_local = j_local_block, j = j_block;
         j_wpt < WORK_PER_THREAD;
         ++j_wpt, ++j_local, ++j) {
      int a_j = k_base + j_local;
      int b_i = k_base + i_local;
      A_local[i_local][j_local] = (i < M && a_j < K) ? A[i * K + a_j] : 0;
      B_local[i_local][j_local] = (j < N && b_i < K) ? B[b_i * N + j] : 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k_local = 0; k_local < CHUNK_SIZE; ++k_local) {
      int a = A_local[i_local][k_local];
      for (int j_wpt = 0, j_local = j_local_block, j = j_block;
           j_wpt < WORK_PER_THREAD;
           ++j_wpt, ++j_local, ++j) {
        result[j_wpt] += a * B_local[k_local][j_local];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (int j_wpt = 0, j = j_block;
       j_wpt < WORK_PER_THREAD;
       ++j_wpt, ++j)
  {
    if (i < M && j < N) {
      C[i * N + j] = result[j_wpt];
    }
  }
}