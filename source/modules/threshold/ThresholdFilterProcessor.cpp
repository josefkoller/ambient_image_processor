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
