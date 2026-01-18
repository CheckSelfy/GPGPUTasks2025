#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* in,
    __global const uint* prefix_sum_accum,
    __global          uint* out,
    unsigned int n,
    unsigned int offset)
{
    uint idx = get_global_id(0);
    if (idx >= n) { return; }

    if ((in[idx] >> offset) & 1) {
        out[prefix_sum_accum[n - 1] - prefix_sum_accum[idx] + idx] = in[idx];
    } else {
        out[prefix_sum_accum[idx] - 1] = in[idx];
    }
}