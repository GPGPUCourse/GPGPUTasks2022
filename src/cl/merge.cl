__kernel void merge(__global const float *as, __global float *res, unsigned int n, unsigned int size_of_part)
{
    const int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const int part_of_array = i / size_of_part;
    const int left_part = part_of_array % 2 == 1 ? part_of_array - 1 : part_of_array;
    const int right_part = part_of_array % 2 == 1 ? part_of_array : part_of_array + 1;

    const int start = left_part * size_of_part;
    const int diagonal = i - start;

    for (int x = 0; x <= diagonal; x++) {
        const int y = diagonal - x;
        if (x >= size_of_part || y >= size_of_part) {
            continue;
        }
        if (as[x + start] <= as[y + size_of_part + start]) {
            res[i] = as[x + start];
            return;
        } else if (x == diagonal) {
            res[i] = as[y + size_of_part + start];
            return;
        }
    }
    res[i] = #TODO
}