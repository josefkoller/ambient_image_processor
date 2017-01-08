/*
    Ambient Image Processor - A tool to perform several imaging tasks
    
    Copyright (C) 2016 Josef Koller

    https://github.com/josefkoller/ambient_image_processor    
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "TGVKDeshadeDownsampledNotMaskedProcessor.h"

#include "ResizeProcessor.h"
#include "TGVKDeshadeProcessor.h"
#include "CudaImageOperationsProcessor.h"

TGVKDeshadeDownsampledNotMaskedProcessor::TGVKDeshadeDownsampledNotMaskedProcessor()
{
}

void TGVKDeshadeDownsampledNotMaskedProcessor::processTGVKL1Cuda(ITKImage input_image,
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
        if(downsampling_factor == 1)
            return original_image;
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

    TGVKDeshadeProcessor::processTGVKL1Cuda(
          downsampled_image,
          lambda,

          order,
          alpha,

          iteration_count,
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
        if(downsampling_factor == 1)
            return original_image;
        return ResizeProcessor::process(original_image,
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
