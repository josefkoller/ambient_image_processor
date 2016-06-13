
#include <fftw3.h>

typedef const unsigned int DimensionSize;

template<typename Pixel>
Pixel* multiply_constant_kernel_launch(Pixel* image1,
                              uint width, uint height, uint depth,
                              Pixel constant);

template<typename Pixel>
void cosine_transform(Pixel* image,
                                 DimensionSize width,
                                 DimensionSize height,
                                 DimensionSize depth,
                                 Pixel* result,
                                 const fftw_r2r_kind kind = FFTW_REDFT10)
{
    fftw_plan plan = fftw_plan_r2r_3d((int)depth, (int) height, (int) width,
                               image, result,
                               kind, kind, kind,
                               FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
    fftw_execute(plan);

    fftw_destroy_plan(plan);
    fftw_cleanup();
}

template<typename Pixel>
void inverse_cosine_transform(Pixel* image,
                                 DimensionSize width,
                                 DimensionSize height,
                                 DimensionSize depth,
                                 Pixel* result)
{
    cosine_transform(image,
      width, height, depth, result, FFTW_REDFT01);
}
