__kernel void merge(__global const float *as, __global float *res, unsigned int n, unsigned int size_of_part)
{
    const int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const int part_of_array = i / size_of_part;
    const int left_part = part_of_array % 2 == 1 ? part_of_array - 1 : part_of_array;
    const int right_part = part_of_array % 2 == 1 ? part_of_array : part_of_array + 1;

    const int start_left_part = left_part * size_of_part;
    const int start_right_part = start_left_part + size_of_part;
    const bool row = (i - start_right_part) >= 0;
    int l = -1, r = size_of_part;

    while (l + 1 < r) {
        int m = (l + r) / 2;
        if (row) {
            if (as[i] < as[start_left_part + m]) {
                r = m;
            } else {
                l = m;
            }
        } else {
            if (as[i] > as[start_right_part + m]) {
                l = m;
            } else {
                r = m;
            }
        }
    }

    res[start_left_part + (i % size_of_part) + l + 1] = as[i];
}