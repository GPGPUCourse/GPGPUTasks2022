#define CHUNK_WIDTH 16
#define CHUNK_HEIGHT 16

__kernel void matrix_transpose(__global const float* src, // rows x cols
                               __global       float* dst, // cols x rows
                               unsigned int rows, unsigned int cols)
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

  int col_src = get_global_id(0);
  int row_src = get_global_id(1);
  int i_unraveled = row_src * cols + col_src;

  int col_chunk_src = get_local_id(0);
  int row_chunk_src = get_local_id(1);
  int i_chunk_unraveled = row_chunk_src * CHUNK_WIDTH + col_chunk_src;

  if (row_src < rows && col_src < cols) {
    int col_chunk_local_dst = row_chunk_src;
    int row_chunk_local_dst = col_chunk_src;
    chunk[row_chunk_local_dst][col_chunk_local_dst] = src[i_unraveled];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  int col_base_dst = row_src - row_chunk_src;
  int row_base_dst = col_src - col_chunk_src;

  int col_chunk_dst = i_chunk_unraveled % CHUNK_HEIGHT;
  int row_chunk_dst = i_chunk_unraveled / CHUNK_HEIGHT;

  int col_dst = col_base_dst + col_chunk_dst;
  int row_dst = row_base_dst + row_chunk_dst;

  if (row_dst < cols && col_dst < rows) {
    dst[row_dst * rows + col_dst] = chunk[row_chunk_src][col_chunk_src];
  }
}

#define SKEWED_CHUNK_WIDTH (CHUNK_HEIGHT + CHUNK_WIDTH - 1)
#define SKEWED_CHUNK_HEIGHT (CHUNK_WIDTH)

__kernel void matrix_transpose_skewed(__global const float* src, // rows x cols
                                      __global       float* dst, // cols x rows
                                      unsigned int rows, unsigned int cols)
{
  __local float chunk[SKEWED_CHUNK_HEIGHT][SKEWED_CHUNK_WIDTH];
  /*
   src:
   a b c d
   e f g h

   local:
   a e . . .
   . b f . .
   . . c g .
   . . . d h

   так и при записи (a b c d),(e g f h) в local,
   и при чтении из него (a e),(b f),(c g),(d h) не будет bank конфликтов
   */

  int col_src = get_global_id(0);
  int row_src = get_global_id(1);
  int i_unraveled = row_src * cols + col_src;

  int col_chunk_src = get_local_id(0);
  int row_chunk_src = get_local_id(1);
  int i_chunk_unraveled = row_chunk_src * CHUNK_WIDTH + col_chunk_src;

  if (row_src < rows && col_src < cols) {
    int row_chunk_local_dst = col_chunk_src;
    int col_chunk_local_dst = row_chunk_src;
    chunk[row_chunk_local_dst][col_chunk_local_dst + row_chunk_local_dst] = src[i_unraveled];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  int col_base_dst = row_src - row_chunk_src;
  int row_base_dst = col_src - col_chunk_src;

  int col_chunk_dst = i_chunk_unraveled % CHUNK_HEIGHT;
  int row_chunk_dst = i_chunk_unraveled / CHUNK_HEIGHT;

  int col_dst = col_base_dst + col_chunk_dst;
  int row_dst = row_base_dst + row_chunk_dst;

  if (row_dst < cols && col_dst < rows) {
    dst[row_dst * rows + col_dst] = chunk[row_chunk_dst][col_chunk_dst + row_chunk_dst];
  }
}
