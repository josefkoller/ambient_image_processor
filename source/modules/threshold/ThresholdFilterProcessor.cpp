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

#include "ThresholdFilterProcessor.h"

#include <itkThresholdImageFilter.h>

ThresholdFilterProcessor::ThresholdFilterProcessor()
{
}



ITKImage ThresholdFilterProcessor::process(
        ITKImage image,
        ImageType::PixelType lower_threshold_value,
        ImageType::PixelType upper_threshold_value,
        ImageType::PixelType outside_pixel_value)
{

    typedef itk::ThresholdImageFilter<ImageType> ThresholdImageFilter;
    ThresholdImageFilter::Pointer filter = ThresholdImageFilter::New();
    filter->SetInput(image.getPointer());
    filter->ThresholdOutside(lower_threshold_value, upper_threshold_value);
    filter->SetOutsideValue(outside_pixel_value);
    filter->Update();

   ImageType::Pointer output = filter->GetOutput();
   output->DisconnectPipeline();

   return ITKImage(output);
}

ITKImage ThresholdFilterProcessor::clamp(ITKImage image,
                                         ITKImage::PixelType lower_threshold_value,
                                         ITKImage::PixelType upper_threshold_value)
{
    auto clamped_image = process(image, lower_threshold_value, 1e6, lower_threshold_value);
    return process(clamped_image, lower_threshold_value, upper_threshold_value, upper_threshold_value);
}
