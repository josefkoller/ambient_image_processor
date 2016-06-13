#ifndef TGVNONPARAMETRICDESHADEPROCESSOR_H
#define TGVNONPARAMETRICDESHADEPROCESSOR_H

#include "ITKImage.h"

#include <functional>

class TGVNonParametricDeshadeProcessor
{
private:
    TGVNonParametricDeshadeProcessor();

    typedef ITKImage::PixelType Pixel;
    typedef const uint DimensionSize;

    template<typename Pixel>
    using CosineTransformCallback = std::function<void(Pixel* image,
                               DimensionSize width, DimensionSize height, DimensionSize depth,
                               Pixel* result, bool is_inverse)>;
public:
    static void performTGVDeshade(
            const ITKImage& input_image,

            const Pixel lambda,
            const ITKImage& mask_image,

            const uint check_iteration_count,
            const Pixel alpha_ratio_step_min,
            const uint final_iteration_count,

            ITKImage& denoised_image,
            ITKImage& shading_image,
            ITKImage& deshaded_image);
};

#endif // TGVNONPARAMETRICDESHADEPROCESSOR_H
