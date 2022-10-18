#define CHUNK_SIZE 16
#define WORK_PER_THREAD 4

__kernel void matrix_multiplication_chunked(__global const int* A,
                                            __global const int* B,
                                            __global       int* C,
                                            unsigned int M, unsigned int K, unsigned int N)
{
  __local int A_local[CHUNK_SIZE][CHUNK_SIZE];
  __local int B_local[CHUNK_SIZE][CHUNK_SIZE];

  int col_local = get_local_id(0);
  int row_local = get_local_id(1);
  int col = get_global_id(0);
  int row = get_global_id(1);

  int result = 0;
  for (int k_base = 0; k_base < K; k_base += CHUNK_SIZE) {
    int a_col = k_base + col_local;
    int b_row = k_base + row_local;
    A_local[row_local][col_local] = (row < M && a_col < K) ? A[row * K + a_col] : 0;
    B_local[row_local][col_local] = (col < N && b_row < K) ? B[b_row * N + col] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k_local = 0; k_local < CHUNK_SIZE; ++k_local) {
      result += A_local[row_local][k_local] * B_local[k_local][col_local];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (row < M && col < N) {
    C[row * N + col] = result;
  }
}

/*
              ....................----....
              ....................----....
              ....................----....
              ....................----....
           K  ....................==#=....
              ....................==#=....
              ....................==#=....
              ....................==#=....
              ....................----....
     K                     N
 .........    ............................
 .........    ............................
 ......... M  ............................
 .........    ............................
 ----####-    ....................==#=....
 ----####-    ....................==#=....
 */

__kernel void matrix_multiplication_multijob(__global const int* A,
                                             __global const int* B,
                                             __global       int* C,
                                             unsigned int M, unsigned int K, unsigned int N)
{
  __local int A_local[CHUNK_SIZE][CHUNK_SIZE];
  __local int B_local[CHUNK_SIZE][CHUNK_SIZE];

  int col_local = get_local_id(0);
  int row_local_block = get_local_id(1) * WORK_PER_THREAD;
  int row_block = get_group_id(1) * CHUNK_SIZE + row_local_block;
  int col = get_global_id(0);

  int result[WORK_PER_THREAD];
  for (int row_wpt = 0; row_wpt < WORK_PER_THREAD; ++row_wpt) {
    result[row_wpt] = 0;
  }

  for (int k_base = 0; k_base < K; k_base += CHUNK_SIZE) {
    for (int row_wpt = 0, row_local = row_local_block, row = row_block;
         row_wpt < WORK_PER_THREAD;
         ++row_wpt, ++row_local, ++row) {
      int a_col = k_base + col_local;
      int b_row = k_base + row_local;
      A_local[row_local][col_local] = (row < M && a_col < K) ? A[row * K + a_col] : 0;
      B_local[row_local][col_local] = (col < N && b_row < K) ? B[b_row * N + col] : 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k_local = 0; k_local < CHUNK_SIZE; ++k_local) {
      int b = B_local[k_local][col_local];
      for (int row_wpt = 0, row_local = row_local_block, row = row_block;
           row_wpt < WORK_PER_THREAD;
           ++row_wpt, ++row_local, ++row) {
        int a = A_local[row_local][k_local];
        result[row_wpt] += a * b;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (int row_wpt = 0, row = row_block;
       row_wpt < WORK_PER_THREAD;
       ++row_wpt, ++row)
  {
    if (row < M && col < N) {
      C[row * N + col] = result[row_wpt];
    }
  }
}