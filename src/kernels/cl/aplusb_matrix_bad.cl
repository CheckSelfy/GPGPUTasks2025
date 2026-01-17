#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__kernel void aplusb_matrix_bad(__global const uint* a,
                     __global const uint* b,
                     __global       uint* c,
                     unsigned int width,
                     unsigned int height)
{
    unsigned gid_x = get_global_id(0);
    unsigned gid_y = get_global_id(1);

    if (gid_x >= width || gid_y >= height) {
        return;
    }

    unsigned index = gid_x * height + gid_y;
    c[index] = a[index] + b[index];
}
