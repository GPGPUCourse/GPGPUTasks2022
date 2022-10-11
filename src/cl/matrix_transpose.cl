#define CHUNK_WIDTH 16
#define CHUNK_HEIGHT 16

__kernel void matrix_transpose(__global const float* src, // m x n
                               __global       float* dst, // n x m
                               unsigned int m, unsigned int n)
{
  __local float chunk[CHUNK_HEIGHT][CHUNK_WIDTH];
  /*
   src:
   a b c
   d e f

   local:
   a d
   b e
   c f
   */

  int j_src = get_global_id(0);
  int i_src = get_global_id(1);
  int j_unraveled = i_src * n + j_src;

  int j_chunk_src = get_local_id(0);
  int i_chunk_src = get_local_id(1);
  int j_chunk_unraveled = i_chunk_src * CHUNK_WIDTH + j_chunk_src;

  if (i_src < m && j_src < n) {
    int j_chunk_local_dst = i_chunk_src;
    int i_chunk_local_dst = j_chunk_src;
    chunk[i_chunk_local_dst][j_chunk_local_dst] = src[j_unraveled];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  int j_base_dst = i_src - i_chunk_src;
  int i_base_dst = j_src - j_chunk_src;

  int j_chunk_dst = j_chunk_unraveled % CHUNK_HEIGHT;
  int i_chunk_dst = j_chunk_unraveled / CHUNK_HEIGHT;

  int j_dst = j_base_dst + j_chunk_dst;
  int i_dst = i_base_dst + i_chunk_dst;

  if (i_dst < n && j_dst < m) {
    dst[i_dst * m + j_dst] = chunk[i_chunk_src][j_chunk_src];
  }
}

#define SKEWED_CHUNK_WIDTH (CHUNK_HEIGHT + CHUNK_WIDTH - 1)
#define SKEWED_CHUNK_HEIGHT (CHUNK_WIDTH)

__kernel void matrix_transpose_skewed(__global const float* src, // m x n
                                      __global       float* dst, // n x m
                                      unsigned int m, unsigned int n)
{
  __local float chunk[SKEWED_CHUNK_HEIGHT][SKEWED_CHUNK_WIDTH + SKEWED_CHUNK_HEIGHT - 1];
  /*
   src:
   a b c
   d e f

   local:
   a d . .
   . b e .
   . . c f
   */

  int j_src = get_global_id(0);
  int i_src = get_global_id(1);
  int j_unraveled = i_src * n + j_src;

  int j_chunk_src = get_local_id(0);
  int i_chunk_src = get_local_id(1);
  int j_chunk_unraveled = i_chunk_src * CHUNK_WIDTH + j_chunk_src;

  if (i_src < m && j_src < n) {
    int i_chunk_local_dst = j_chunk_src;
    int j_chunk_local_dst = i_chunk_src;
    chunk[i_chunk_local_dst][j_chunk_local_dst + i_chunk_local_dst] = src[j_unraveled];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  int j_base_dst = i_src - i_chunk_src;
  int i_base_dst = j_src - j_chunk_src;

  int j_chunk_dst = j_chunk_unraveled % CHUNK_HEIGHT;
  int i_chunk_dst = j_chunk_unraveled / CHUNK_HEIGHT;

  int j_dst = j_base_dst + j_chunk_dst;
  int i_dst = i_base_dst + i_chunk_dst;

  if (i_dst < n && j_dst < m) {
    dst[i_dst * m + j_dst] = chunk[i_chunk_dst][j_chunk_dst + i_chunk_dst];
  }
}
