#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    uint y = get_global_id(1); 
    uint x = get_global_id(0); 
    uint local_y = get_local_id(1); 
    uint local_x = get_local_id(0); 

    // prevent bank conflicts
    __local float buffer[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    
    if (y < h && x < w) {
        buffer[local_y][local_x] = matrix[y * w + x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint tx = get_group_id(1) * GROUP_SIZE_Y + local_y; 
    uint ty = get_group_id(0) * GROUP_SIZE_X + local_x; 

    if (ty < w && tx < h) {
        transposed_matrix[ty * h + tx] = buffer[local_y][local_x];
    }
}
