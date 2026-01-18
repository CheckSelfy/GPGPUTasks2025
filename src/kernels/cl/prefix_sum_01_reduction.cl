#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* pow2_sum, // contains n values
    __global       uint* next_pow2_sum, // will contain (n+1)/2 values
    uint n)
{
    uint idx = get_global_id(0);

    // if (idx * 2 >  n) {
    //     printf("error, idx=%d, n=%d\n", idx, n);
    // }

    if (idx * 2 + 1 < n) { 
        next_pow2_sum[idx] = pow2_sum[idx  * 2] + pow2_sum[idx * 2 + 1];
    } else if (idx * 2 < n) {
        next_pow2_sum[idx] = pow2_sum[idx  * 2];
    } else {
        next_pow2_sum[idx] = 0;
    }
}
