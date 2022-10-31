__kernel void psum_fill_zero(__global uint* as, uint n) {
    size_t id = get_global_id(0);
    if (id < n) {
        as[id] = 0;
    }
}

__kernel void psum_reduce_adjacent(__global uint* in, __global uint* out, uint n) {
    size_t id = get_global_id(0);
    if (id * 2 < n) {
        out[id] = in[id * 2] + ((id * 2 + 1 < n) ? in[id * 2 + 1] : 0);
    }
}

__kernel void psum_reduce(__global uint* as, __global uint* bs, uint n, uint step) {
    size_t id = get_global_id(0);
    if (id < n && ((id + 1) & ((size_t)1 << step))) {
        bs[id] += as[((id + 1) >> step) - 1];
    }
}
