#define WORK_GROUP_SIZE 128
#define LOCALLY_SORTABLE_SIZE_3
// #define LOCALLY_SORTABLE_SIZE 3

#define CAT(X,Y) X##_##Y   //concatenate words


#ifdef LOCALLY_SORTABLE_SIZE_2
#define LOCALLY_SORTABLE_SIZE 2
void sort_small(__local float* a_ptr) {
    float a[2] = { a_ptr[0], a_ptr[1] };

    if (a[0] > a[1]) {
        a_ptr[0] = a[1];
        a_ptr[1] = a[0];
    }
}
#endif


#ifdef LOCALLY_SORTABLE_SIZE_3
#define LOCALLY_SORTABLE_SIZE 3
void sort_small(__local float* a_ptr) {
  float a[3] = { a_ptr[0], a_ptr[1], a_ptr[2] };

  if (a[0] > a[1]) {
    if (a[1] > a[2]) {
      a_ptr[0] = a[2];
      a_ptr[2] = a[0];
    } else {
      a_ptr[0] = a[1];
      if (a[0] > a[2]) {
        a_ptr[1] = a[2];
        a_ptr[2] = a[0];
      } else {
        a_ptr[1] = a[0];
      }
    }
  } else if (a[1] > a[2]) {
    a_ptr[2] = a[1];
    if (a[0] > a[2]) {
      a_ptr[0] = a[2];
      a_ptr[1] = a[0];
    } else {
      a_ptr[1] = a[2];
    }
  }
}

#define MERGE_SMALL(len_1)                                                        \
  void merge_small##_##len_1 (__local float* a_ptr) {                             \
    unsigned int len_2 = 6 - len_1;                                               \
                                                                                  \
    float a[6] = { a_ptr[0], a_ptr[1], a_ptr[2], a_ptr[3], a_ptr[4], a_ptr[5] };  \
                                                                                  \
    int j = 0;                                                                    \
    for (int i = 0; i < len_1; ++i) {                                             \
      /*    Тут 6 раз вложена в себя такая конструкция:   */                      \
      /*    if (j < len_2 && a[i] > a[len_1 + j]) {       */                      \
      /*      a_ptr[i + j] = a[len_1 + j];                */                      \
      /*      ++j;                                        */                      \
      /*    }                                             */                      \
                                                                                  \
      if (j < len_2 && a[i] > a[len_1 + j]) {                                     \
        a_ptr[i + j] = a[len_1 + j];                                              \
        ++j;                                                                      \
                                                                                  \
        if (j < len_2 && a[i] > a[len_1 + j]) {                                   \
          a_ptr[i + j] = a[len_1 + j];                                            \
          ++j;                                                                    \
                                                                                  \
          if (j < len_2 && a[i] > a[len_1 + j]) {                                 \
            a_ptr[i + j] = a[len_1 + j];                                          \
            ++j;                                                                  \
                                                                                  \
            if (j < len_2 && a[i] > a[len_1 + j]) {                               \
              a_ptr[i + j] = a[len_1 + j];                                        \
              ++j;                                                                \
                                                                                  \
              if (j < len_2 && a[i] > a[len_1 + j]) {                             \
                a_ptr[i + j] = a[len_1 + j];                                      \
                ++j;                                                              \
                                                                                  \
                if (j < len_2 && a[i] > a[len_1 + j]) {                           \
                  a_ptr[i + j] = a[len_1 + j];                                    \
                  ++j;                                                            \
                }                                                                 \
              }                                                                   \
            }                                                                     \
          }                                                                       \
        }                                                                         \
      }                                                                           \
      a_ptr[i + j] = a[i];                                                        \
    }                                                                             \
    /* этот цикл ничего не делал бы         */                                    \
    /*  for (; j < len_2; ++j) {            */                                    \
    /*    a_ptr[len_1 + j] = a[len_1 + j];  */                                    \
    /*  }                                   */                                    \
  }

MERGE_SMALL(1)
MERGE_SMALL(2)
MERGE_SMALL(3)
MERGE_SMALL(4)
MERGE_SMALL(5)

void merge_small(__local float* a_ptr, int len_1) {
  switch (len_1) {
  case 1:
    merge_small_1(a_ptr);
    break;
  case 2:
    merge_small_2(a_ptr);
    break;
  case 3:
    merge_small_3(a_ptr);
    break;
  case 4:
    merge_small_4(a_ptr);
    break;
  case 5:
    merge_small_5(a_ptr);
    break;
  default:
    break;
  }
}

void merge_small_unknown(__local float* a_ptr) {
  for (int i = 0; i < 5; ++i) {
    if (a_ptr[i] > a_ptr[i + 1]) {
      merge_small(a_ptr, i + 1);
      break;
    }
  }
}
#endif

// #define WORK_PER_THREAD (2 * LOCALLY_SORTABLE_SIZE) == 6
#define LOCAL_SIZE ((LOCALLY_SORTABLE_SIZE * 2) * WORK_GROUP_SIZE)

void swap_local(__local float* a, __local float* b) {
  float tmp = *a;
  *a = *b;
  *b = tmp;
}

void swap_global(__global float* a, __global float* b) {
  float tmp = *a;
  *a = *b;
  *b = tmp;
}

void reverse_local(__local float* a, unsigned int len) {
  if (len > 0) {
    for (unsigned int i = 0; i < len - 1 - i; ++i) {
      swap_local(a + i, a + (len - 1 - i));
    }
  }
}

void reverse_global(__global float* a, unsigned int len) {
  if (len > 0) {
    for (unsigned int i = 0; i < len - 1 - i; ++i) {
      swap_global(a + i, a + (len - 1 - i));
    }
  }
}

void swap_local_arrays(__local float* a, unsigned int len_1, unsigned int len_2) {
  reverse_local(a, len_1 + len_2);
  reverse_local(a, len_2);
  reverse_local(a + len_2, len_1);
}

void swap_global_arrays(__global float* a, unsigned int len_1, unsigned int len_2) {
  reverse_global(a, len_1 + len_2);
  reverse_global(a, len_2);
  reverse_global(a + len_2, len_1);
}

void load(__global float* a, __local float* local_a, unsigned int n,
          unsigned int global_id, unsigned int group_id, unsigned int local_id,
          int download_flag) {
  unsigned int group_offset = group_id * LOCAL_SIZE;

  for (unsigned int mergeable_i = 0, row_offset = 0;
       mergeable_i < 2;
       ++mergeable_i)
  {
    for (unsigned int sortable_i = 0;
         sortable_i < LOCALLY_SORTABLE_SIZE;
         ++sortable_i, row_offset += WORK_GROUP_SIZE)
    {
      unsigned int i = row_offset + local_id;
      if (download_flag) {
        local_a[i] = global_id < n ? a[group_offset + i] : INFINITY;
      } else if (global_id < n) {
        a[group_offset + i] = local_a[i];
      }
    }
  }
}

void download(__global float* a, __local float* local_a, unsigned int n,
              unsigned int global_id, unsigned int group_id, unsigned int local_id) {

  load(a, local_a, n, global_id, group_id, local_id, true);
  barrier(CLK_LOCAL_MEM_FENCE);
}

void upload(__global float* a, __local float* local_a, unsigned int n,
                  unsigned int global_id, unsigned int group_id, unsigned int local_id) {
  barrier(CLK_LOCAL_MEM_FENCE);
  load(a, local_a, n, global_id, group_id, local_id, false);
}

__kernel void merge_path_global(__global float* a,
                                unsigned int n, unsigned int merged_src_len)
{
  int debug_flag = false;

  __local float local_a[LOCAL_SIZE];

  unsigned int local_id = get_local_id(0);
  unsigned int global_id = get_global_id(0);
  unsigned int group_id = get_group_id(0);

  if (global_id == 0) {
    if (debug_flag) {
      printf("Global, merged_src_len=%u\n", merged_src_len);
    }
    for (unsigned int offset = 0; offset < n; offset += merged_src_len * 2) {
      __global float *first = a + offset;
      unsigned int len_1 = min(merged_src_len, n - offset);

      __global float *second = first + len_1;
      unsigned int len_2 = min(merged_src_len, n - offset - len_1);

      if (debug_flag) {
        printf("offset=%u, len_1=%u, len_2=%u\n", offset, len_1, len_2);
      }

      while (len_1 > 0 && len_2 > 0) {
        unsigned int diag = min(LOCALLY_SORTABLE_SIZE * 2u, len_1 + len_2);

        unsigned int len_1_1;
        unsigned int len_1_2;
        unsigned int len_2_1;
        unsigned int len_2_2;

        unsigned int l = len_2 < diag ? diag - len_2 : 0;
        unsigned int r = min(diag, len_1);

        if (l == len_1 || first[l] > second[diag - l - 1]) {
          len_1_1 = l;
        } else {
          while (l + 1 < r) {
            unsigned int mid = (l + r) / 2;
            if (first[mid] <= second[diag - mid - 1]) {
              l = mid;
            } else {
              r = mid;
            }
          }
          len_1_1 = r;
        }

        len_2_1 = diag - len_1_1;
        len_1_2 = len_1 - len_1_1;
        len_2_2 = len_2 - len_2_1;

        if (len_1_2 > 0 && len_2_1 > 0) {
          swap_global_arrays(first + len_1_1, len_1_2, len_2_1);
          // было  len_1_1 len_1_2 len_2_1 len_2_2
          // стало len_1_1 len_2_1 len_1_2 len_2_2
        }

        if (len_1 < len_1_1) {
          len_1 = 0;
        } else {
          first += diag;
          len_1 -= len_1_1;
        }

        if (len_2 < len_2_1) {
          len_2 = 0;
        } else {
          second += len_2_1;
          len_2 -= len_2_1;
        }
      } // end while
    } // end for
  } // end if global_id == 0
}

__kernel void merge_global(__global float* a,
                         unsigned int n)
{
  __local float local_a[LOCAL_SIZE];

  unsigned int local_id = get_local_id(0);
  unsigned int global_id = get_global_id(0);
  unsigned int group_id = get_group_id(0);

  download(a, local_a, n, global_id, group_id, local_id);
  merge_small_unknown(local_a + local_id * (LOCALLY_SORTABLE_SIZE * 2));
  upload(a, local_a, n, global_id, group_id, local_id);
}

__kernel void merge_local(__global float* a,
                         unsigned int n)
{
  int debug_flag = false;

  __local float local_a[LOCAL_SIZE];

  unsigned int local_id = get_local_id(0);
  unsigned int global_id = get_global_id(0);
  unsigned int group_id = get_group_id(0);

  download(a, local_a, n, global_id, group_id, local_id);

  if (debug_flag && global_id == 0) {
    printf("Init 0\n");
    for (unsigned int i = 0; i < 24; ++i) {
      printf("%.0f ", local_a[i]);
    }
    printf("\n\n");
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  sort_small(local_a + local_id * (LOCALLY_SORTABLE_SIZE * 2));
  sort_small(local_a + local_id * (LOCALLY_SORTABLE_SIZE * 2) + LOCALLY_SORTABLE_SIZE);

  barrier(CLK_LOCAL_MEM_FENCE);

//  if (debug_flag && global_id == 0) {
//    printf("Sorted 0 \n");
//    for (unsigned int i = 0; i < 24; ++i) {
//      printf("%.0f ", local_a[i]);
//    }
//    printf("\n\n");
//  }
//  barrier(CLK_LOCAL_MEM_FENCE);
//
//  // теперь каждый поток будет мержить по два отрезка длиной LOCALLY_SORTABLE_SIZE
//  __local unsigned int first_arr_lengths[WORK_GROUP_SIZE];
//
//  for (unsigned int scale_factor = 0, merged_src_len = LOCALLY_SORTABLE_SIZE;
//       merged_src_len < LOCAL_SIZE;
//       ++scale_factor, merged_src_len *= 2)
//  {
//    first_arr_lengths[local_id] = merged_src_len; // todo move down
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    for (unsigned int factor_i = scale_factor, merged_len = merged_src_len * 2;
//        factor_i > 0;
//         --factor_i, merged_len /= 2)
//    {
//      if (debug_flag && global_id == 0) {
//        printf("Inner cycle. scale=%u, factor=%u, merged_src_len=%u\n",
//               scale_factor, factor_i, merged_src_len);
//        //        for (unsigned int i = 0; i < 12; ++i) {
//        //          printf("%.0f ", local_a[i]);
//        //        }
//        //        printf("\n\n");
//      }
//
//      if (local_id % (1 << factor_i) == 0) {
//        unsigned int diag = merged_src_len;
//
//        unsigned int len_1 = first_arr_lengths[local_id];
//        __local float *first = local_a;
//
//        unsigned int len_2 = merged_len - len_1;
//        __local float *second = first + len_1;
//
////        if (debug_flag) {
////          printf("len_1=%u, len_2=%u\n", len_1, len_2);
////        }
//
//        unsigned int len_1_1;
//        unsigned int len_1_2;
//        unsigned int len_2_1;
//        unsigned int len_2_2;
//
//        unsigned int l = len_2 < diag ? diag - len_2 : 0;
//        unsigned int r = min(diag, len_1);
//
//        if (len_1 == 0 || (len_2 != 0 && first[0] > second[len_2 - 1])) {
//          len_1_1 = 0;
//        } else if (len_2 == 0 ||
//                   (len_1 != 0 && first[len_1 - 1] <= second[0])) {
//          len_1_1 = len_1;
//        } else if (l == len_1 || first[l] > second[diag - l - 1]) {
//          len_1_1 = l;
//        } else {
//          while (l + 1 < r) {
//            unsigned int mid = (l + r) / 2;
//            if (first[mid] <= second[diag - mid - 1]) {
//              l = mid;
//            } else {
//              r = mid;
//            }
//          }
//          len_1_1 = r;
//        }
//
//        len_1_2 = len_1 - len_1_1;
//        len_2_1 = diag - len_1_1;
//        len_2_2 = len_2 - len_2_1;
//
//        if (len_1_2 > 0 && len_2_1 > 0) {
//          swap_local_arrays(first + len_1_1, len_1_2, len_2_1);
//          // было  len_1_1 len_1_2 len_2_1 len_2_2
//          // стало len_1_1 len_2_1 len_1_2 len_2_2
//        }
//
//        first += diag;
//        len_1 -= len_1_1;
//
//        second += len_2_1;
//        len_2 -= len_2_1;
//
//        first_arr_lengths[local_id] = len_1_1;
//        first_arr_lengths[local_id + (1 << factor_i) / 2] = len_1_2;
//        barrier(CLK_LOCAL_MEM_FENCE);
//
//
////        unsigned int len_1 = first_arr_lengths[local_id];
////        unsigned int len_2 = merged_len - len_1;
////        unsigned int diag = merged_len / 2;
////
////        __local float *first = local_a + (local_id / (1 << factor_i)) * merged_len;
////        __local float *second = first + len_1;
////
////        unsigned int len_1_1;
////        unsigned int len_1_2;
////        unsigned int len_2_1;
////        unsigned int len_2_2;
////
////        if (len_1 == 0 || (len_2 != 0 && first[0] > second[len_2 - 1])) {
////          len_1_1 = 0;
////        } else if (len_2 == 0 ||
////                   (len_1 != 0 && first[len_1 - 1] <= second[0])) {
////          len_1_1 = len_1;
////        } else {
////          unsigned int l = len_2 < diag ? diag - len_2 : 0;
////          unsigned int r = min(diag, len_1);
////
////          while (l + 1 < r) {
////            unsigned int mid = (l + r) / 2;
////            if (first[mid] <= second[diag - 1 - mid]) {
////              l = mid;
////            } else {
////              r = mid;
////            }
////          }
////
////          len_1_1 = r;
////        }
//////        len_1_2 = len_1 - len_1_1;
//////        len_2_1 = diag - len_1_1;
//////        len_2_2 = len_2 - len_2_1;
//////
//////        swap_local_arrays(first + len_1_1, len_1_2, len_2_1);
//////        // было  len_1_1 len_1_2 len_2_1 len_2_2
//////        // стало len_1_1 len_2_1 len_1_2 len_2_2
////
////        first_arr_lengths[local_id] = len_1_1;
////        first_arr_lengths[local_id + (1 << factor_i) / 2] = len_1_2;
//      }
//      barrier(CLK_LOCAL_MEM_FENCE);
//
//      if (debug_flag && global_id == 0) {
//        printf("After binsearch\n");
//        for (unsigned int i = 0; i < 4; ++i) {
//          printf("%u ", first_arr_lengths[i]);
//        }
//        printf("\n");
//        for (unsigned int i = 0; i < 24; ++i) {
//          printf("%.0f ", local_a[i]);
//        }
//        printf("\n\n");
//      }
//    }
//
    barrier(CLK_LOCAL_MEM_FENCE);
    merge_small_unknown(local_a + local_id * (LOCALLY_SORTABLE_SIZE * 2)/*,
                first_arr_lengths[local_id]*/);
    barrier(CLK_LOCAL_MEM_FENCE);
//
//    if (debug_flag && global_id == 0) {
//      printf("After merge\n");
//      for (unsigned int i = 0; i < 24; ++i) {
//        printf("%.0f ", local_a[i]);
//      }
//      printf("\n");
//
//      int ok_flag = true;
//      for (int i = 0; i < LOCAL_SIZE; i += merged_src_len * 2) {
//        for (int j = 0; j < merged_src_len * 2 - 1 && j < (LOCAL_SIZE - i - 1); ++j) {
//          ok_flag &= (local_a[j] <= local_a[j + 1]);
//        }
//      }
//      if (!ok_flag) {
//        printf("bad merge\n");
//      }
//      printf("\n");
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);
//  }

  upload(a, local_a, n, global_id, group_id, local_id);
}