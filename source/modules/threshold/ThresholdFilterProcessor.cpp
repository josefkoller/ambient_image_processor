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
