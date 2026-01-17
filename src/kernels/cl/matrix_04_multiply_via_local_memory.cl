#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint local_x = get_local_id(0);
    uint local_y = get_local_id(1);
    float sum=0;
    if (x >= w || y >= h) { return; }

    __local float a_cols[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float b_rows[GROUP_SIZE_Y][GROUP_SIZE_X];

    for (uint i = 0; i < ((k + GROUP_SIZE_X - 1) / GROUP_SIZE_X); i++) {
        uint col  = i * GROUP_SIZE_X + local_x;
        uint row = i * GROUP_SIZE_Y + local_y;

        if (y < h && col < k) {
            a_cols[local_y][local_x] = a[y * k + col];
        } else {
            a_cols[local_y][local_x] = 0;
        }

        if (x < w && row < k) {
            b_rows[local_y][local_x] = b[row * w + x];
        } else {
            b_rows[local_y][local_x] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint idx = 0; idx < GROUP_SIZE_X; idx++) {
            sum += a_cols[local_y][idx] * b_rows[idx][local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[y * w + x] = sum;

}
