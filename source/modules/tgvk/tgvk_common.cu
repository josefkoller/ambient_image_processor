#ifndef TGV_K_COMMON
#define TGV_K_COMMON

#include "cuda.h"

template<typename Pixel>
__global__ void tgvk_kernel_part5(
        Pixel* r2_x, Pixel* r2_y, Pixel* r2_z,
        Pixel* r2_xy, Pixel* r2_xz, Pixel* r2_yz,

        Pixel* r_x, Pixel* r_y, Pixel* r_z,
        Pixel* r_xy, Pixel* r_xz, Pixel* r_yz,

        Pixel* w_x, Pixel* w_y, Pixel* w_z,
        Pixel* w_xy, Pixel* w_xz, Pixel* w_yz,

        Pixel* w_previous_x, Pixel* w_previous_y, Pixel* w_previous_z,
        Pixel* w_previous_xy, Pixel* w_previous_xz, Pixel* w_previous_yz,

        Pixel* w_prime_x, Pixel* w_prime_y, Pixel* w_prime_z,
        Pixel* w_prime_xy, Pixel* w_prime_xz, Pixel* w_prime_yz,

        const Pixel sigma, const Pixel alpha2,
        const uint width, const uint height, const uint depth) {

    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= width * height * depth)
        return;

    r_x[index] += sigma * (r2_x[index] - w_prime_x[index]);
    r_y[index] += sigma * (r2_y[index] - w_prime_y[index]);
    r_xy[index] += sigma * (r2_xy[index] - w_prime_xy[index]);

    if(depth > 1) {
        r_z[index] += sigma * (r2_z[index] - w_prime_z[index]);
        r_xz[index] += sigma * (r2_xz[index] - w_prime_xz[index]);
        r_yz[index] += sigma * (r2_yz[index] - w_prime_yz[index]);
    }

    Pixel normalization =
            r_x[index] * r_x[index] +
            r_y[index] * r_y[index] +
            2 * r_xy[index] * r_xy[index];

    if(depth > 1)
        normalization += r_z[index] * r_z[index] +
                2 * r_xz[index] * r_xz[index] +
                2 * r_yz[index] * r_yz[index];

    normalization = fmax(1, sqrt(normalization) / alpha2);

    r_x[index] /= normalization;
    r_y[index] /= normalization;
    r_xy[index] /= normalization;
    if(depth > 1) {
        r_z[index] /= normalization;
        r_xz[index] /= normalization;
        r_yz[index] /= normalization;

        w_previous_z[index] = w_z[index];
        w_previous_xz[index] = w_xz[index];
        w_previous_yz[index] = w_yz[index];
    }

    w_previous_x[index] = w_x[index];
    w_previous_y[index] = w_y[index];
    w_previous_xy[index] = w_xy[index];
}
#endif // TGV_K_COMMON
