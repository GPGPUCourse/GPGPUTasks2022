__kernel void merge(__global float *a, __global float *result,
                    unsigned int n, unsigned int merged_src_len)
{
  const unsigned int global_id = get_global_id(0);

  if (global_id >= n)
    return;

  unsigned int offset = global_id / (merged_src_len * 2) * merged_src_len * 2;
  unsigned int diag = global_id - offset + 1;

  unsigned int len_1 = min(merged_src_len, n - offset);
  unsigned int len_2 = min(merged_src_len, n - offset - len_1);
  __global float *first = a + offset;
  __global float *second = first + len_1;

  unsigned int l = len_2 < (diag - 1) ? (diag - 1) - len_2 : 0;
  unsigned int r = min(diag - 1, len_1);

  while (l < r) {
    unsigned int mid = l + (r - l) / 2;
    if (first[mid] <= second[(diag - 1) - mid - 1]) {
      l = mid + 1;
    } else {
      r = mid;
    }
  }

  result[global_id] = (r < len_1 && (diag - r - 1 >= len_2 || first[r] <= second[diag - r - 1]))
      ? first[r]
      : second[diag - r - 1];
}
