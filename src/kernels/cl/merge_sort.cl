#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* in,
    __global       uint* out,
                   int   n,
                   int   chunk_size)
{
    const uint i = get_global_id(0);
    if (i >= n) return;

    int block_size = chunk_size * 2;
    int block_start = i - i % block_size;
    int mid = block_start + chunk_size;
    int offset = i % chunk_size;

    if (mid >= n) {
        out[i] = in[i];
        return;
    }

    bool from_first = i < mid;
    int other_start = from_first ? mid : block_start;

    int l = -1, r = chunk_size;
    while (r - l > 1) {
        int m = (l + r) / 2;
        int idx = other_start + m;
        if (idx < n &&
            ((in[idx] < in[i]) || (in[idx] <= in[i] && from_first))) {
            l = m;
        } else {
            r = m;
        }
    }

    out[block_start + offset + r] = in[i];
}