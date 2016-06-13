#include "TGVNonParametricDeshadeProcessor.h"

#include "cosine_transform.h"
#include <functional>

template<typename Pixel>
void tgv2_l1_non_parametric_deshade_launch(
    Pixel* f_host,
    DimensionSize width, DimensionSize height, DimensionSize depth,
    const Pixel lambda,
    Pixel* mask,

    const uint check_iteration_count,
    const Pixel alpha_step_minimum,
    const uint final_iteration_count,

    TGVNonParametricDeshadeProcessor::CosineTransformCallback<Pixel> cosine_transform_callback,
    Pixel** denoised_host,
    Pixel** shading_host,
    Pixel** deshaded_host);

TGVNonParametricDeshadeProcessor::TGVNonParametricDeshadeProcessor()
{
}

void TGVNonParametricDeshadeProcessor::performTGVDeshade(
        const ITKImage& input_image,
        const Pixel lambda,
        const ITKImage& mask_image,

        const uint check_iteration_count,
        const Pixel alpha_step_minimum,
        const uint final_iteration_count,


        ITKImage& denoised_image,
        ITKImage& shading_image,
        ITKImage& deshaded_image)
{
    CosineTransformCallback<Pixel> cosine_transform_callback = [](Pixel* image,
            DimensionSize width, DimensionSize height, DimensionSize depth,
            Pixel* result, bool is_inverse = false) {
        if(is_inverse)
            inverse_cosine_transform(image, width, height, depth, result);
        else
            cosine_transform(image, width, height, depth, result);
    };

    auto f = input_image.cloneToPixelArray();
    auto mask = mask_image.cloneToPixelArray();
    Pixel* denoised_pixels, *shading_pixels, *deshaded_pixels;

    tgv2_l1_non_parametric_deshade_launch(f, input_image.width, input_image.height, input_image.depth,
        lambda, mask,
        check_iteration_count, alpha_step_minimum, final_iteration_count,
        cosine_transform_callback,
        &denoised_pixels, &shading_pixels, &deshaded_pixels);

    delete[] f;
    if(mask != nullptr)
        delete[] mask;

     denoised_image = ITKImage(input_image.width, input_image.height, input_image.depth, denoised_pixels);
     shading_image = ITKImage(input_image.width, input_image.height, input_image.depth, shading_pixels);
     deshaded_image = ITKImage(input_image.width, input_image.height, input_image.depth, deshaded_pixels);

     delete[] denoised_pixels;
     delete[] shading_pixels;
     delete[] deshaded_pixels;
}
