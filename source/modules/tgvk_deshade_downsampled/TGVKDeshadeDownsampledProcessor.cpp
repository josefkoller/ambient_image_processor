#include "TGVKDeshadeDownsampledProcessor.h"

#include "ResizeProcessor.h"
#include "TGVKDeshadeMaskedProcessor.h"
#include "CudaImageOperationsProcessor.h"

TGVKDeshadeDownsampledProcessor::TGVKDeshadeDownsampledProcessor()
{
}

void TGVKDeshadeDownsampledProcessor::processTGVKL1Cuda(ITKImage input_image,
                             const Pixel downsampling_factor,
                             const Pixel lambda,

                             const uint order,
                             const Pixel* alpha,

                             const uint iteration_count,
                             const ITKImage& mask,
                             const bool set_negative_values_to_zero,
                             const bool add_background_back,

                             const uint paint_iteration_interval,
                             IterationFinishedThreeImages iteration_finished_callback,

                             ITKImage& denoised_image,
                             ITKImage& shading_image,
                             ITKImage& deshaded_image,
                             ITKImage& div_v_image,
                             const bool calculate_div_v)
{
    ResizeProcessor::InterpolationMethod interpolation_method = ResizeProcessor::InterpolationMethod::BSpline3;
    auto downsample = [=](ITKImage original_image) {
        return ResizeProcessor::process(original_image, downsampling_factor, interpolation_method);
    };

    auto downsampled_image = downsample(input_image);

    ITKImage downsampled_mask = mask;
    if(!mask.isNull())
        downsampled_mask = downsample(mask);

    ITKImage downsampled_denoised_image;
    ITKImage downsampled_shading_image;
    ITKImage downsampled_deshaded_image;
    ITKImage downsampled_div_v_image;

    const int cuda_block_dimension = -1;
    TGVKDeshadeMaskedProcessor::processTGVKL1Cuda(
          downsampled_image,
          lambda,

          order,
          alpha,

          iteration_count,
          cuda_block_dimension,
          downsampled_mask,
          set_negative_values_to_zero,
          add_background_back,

          paint_iteration_interval,
          iteration_finished_callback,

          downsampled_denoised_image,
          downsampled_shading_image,
          downsampled_deshaded_image,
          downsampled_div_v_image,
          calculate_div_v);

    auto upsample = [=](ITKImage original_image) {
        return ResizeProcessor::process(original_image,
                                        1 / downsampling_factor,
                                        input_image.width, input_image.height, input_image.depth,
                                        interpolation_method);
    };

    denoised_image = upsample(downsampled_denoised_image);
    shading_image = upsample(downsampled_shading_image);

    deshaded_image = CudaImageOperationsProcessor::subtract(input_image, shading_image);
    if(!mask.isNull())
        deshaded_image = CudaImageOperationsProcessor::multiply(deshaded_image, mask);

    if(add_background_back && !mask.isNull())
    {
        auto background_mask = CudaImageOperationsProcessor::invert(mask);
        auto background = CudaImageOperationsProcessor::multiply(input_image, background_mask);
        deshaded_image = CudaImageOperationsProcessor::add(deshaded_image, background);
    }


    if(set_negative_values_to_zero)
        deshaded_image = CudaImageOperationsProcessor::clamp_negative_values(deshaded_image, 0);

    if(calculate_div_v)
        div_v_image = upsample(downsampled_div_v_image);
}
