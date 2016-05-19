#include "UnsharpMaskingProcessor.h"

#include <itkDiscreteGaussianImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkAddImageFilter.h>

UnsharpMaskingProcessor::UnsharpMaskingProcessor()
{
}


ITKImage UnsharpMaskingProcessor::process(ITKImage source,
                                                 uint kernel_size, float kernel_sigma, float factor)
{
    typedef ITKImage::InnerITKImage Image;
    auto image = source.getPointer();

    typedef itk::DiscreteGaussianImageFilter<Image, Image> BlurFilter;
    BlurFilter::Pointer blur_filter = BlurFilter::New();
    blur_filter->SetInput(image);
    blur_filter->SetMaximumKernelWidth(kernel_size);
    blur_filter->SetVariance(kernel_sigma);
    blur_filter->Update();
    Image::Pointer low_pass_image = blur_filter->GetOutput();
    low_pass_image->DisconnectPipeline();

    typedef itk::SubtractImageFilter<Image, Image, Image> SubtractFilter;
    SubtractFilter::Pointer subtract_filter = SubtractFilter::New();
    subtract_filter->SetInput1(image);
    subtract_filter->SetInput2(low_pass_image);
    subtract_filter->Update();
    Image::Pointer high_pass_image = subtract_filter->GetOutput();
    high_pass_image->DisconnectPipeline();

    typedef itk::MultiplyImageFilter<Image, Image, Image> MultiplyFilter;
    MultiplyFilter::Pointer multiply_filter = MultiplyFilter::New();
    multiply_filter->SetInput1(high_pass_image);
    multiply_filter->SetConstant2(factor);
    multiply_filter->Update();
    Image::Pointer high_pass_image_scaled = multiply_filter->GetOutput();
    high_pass_image_scaled->DisconnectPipeline();

    typedef itk::AddImageFilter<Image, Image, Image> AddFilter;
    AddFilter::Pointer add_filter = AddFilter::New();
    add_filter->SetInput1(image);
    add_filter->SetInput2(high_pass_image_scaled);
    add_filter->Update();
    Image::Pointer output = add_filter->GetOutput();
    output->DisconnectPipeline();

    return ITKImage(output);
}
