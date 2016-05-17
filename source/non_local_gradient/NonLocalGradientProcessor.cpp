#include "NonLocalGradientProcessor.h"

#include <itkImageRegionIteratorWithIndex.h>

#include <iostream>

typedef NonLocalGradientProcessor::Image::PixelType FloatType;

extern "C" FloatType* non_local_gradient_kernel_launch(
        FloatType* source,
    uint source_width, uint source_height, FloatType* kernel, uint kernel_size);

NonLocalGradientProcessor::NonLocalGradientProcessor()
{
}

NonLocalGradientProcessor::Image::PixelType* NonLocalGradientProcessor::createKernel(
        uint kernel_size,
        Image::PixelType kernel_sigma)
{
    Image::PixelType* kernel = new Image::PixelType[kernel_size * kernel_size];
    uint kernel_center = std::floor(kernel_size / 2.0f);

    Image::PixelType kernel_value_sum = 0;
    for(uint y = 0; y < kernel_size; y++)
    {
        for(uint x = 0; x < kernel_size; x++)
        {
            uint xr = x - kernel_center;
            uint yr = y - kernel_center;

            Image::PixelType radius = std::sqrt(xr*xr + yr*yr);
            Image::PixelType value = std::exp(-radius*radius / kernel_sigma);

            uint i = x + y * kernel_size;
            kernel[i] = value;
            kernel_value_sum += value;
        }
    }

    for(uint y = 0; y < kernel_size; y++)
    {
        for(uint x = 0; x < kernel_size; x++)
        {
            uint i = x + y * kernel_size;
            Image::PixelType value = kernel[i];
            value /= kernel_value_sum;

//            std::cout << "kernel value1: " << value << std::endl;

            kernel[i] = value;
        }
    }

    return kernel;
}

NonLocalGradientProcessor::Image::Pointer NonLocalGradientProcessor::process(Image::Pointer source,
                              uint kernel_size,
                              Image::PixelType kernel_sigma)
{
    Image::PixelType* source_data = source->GetBufferPointer();
    Image::PixelType* kernel_data = createKernel(kernel_size, kernel_sigma);

    Image::SizeType source_size = source->GetLargestPossibleRegion().GetSize();
    uint source_width = source_size[0];
    uint source_height = source_size[1];

    Image::PixelType* destination_data = non_local_gradient_kernel_launch(source_data,
        source_width, source_height, kernel_data, kernel_size);

    delete[] kernel_data;

    Image::Pointer destination = Image::New();
    destination->SetRegions(source_size);
    destination->Allocate();
    itk::ImageRegionIteratorWithIndex<Image> destination_iterator(destination,
        destination->GetLargestPossibleRegion());
    while(!destination_iterator.IsAtEnd())
    {
        Image::IndexType index = destination_iterator.GetIndex();
        int i = index[0] + index[1] * source_width;

        destination_iterator.Set(destination_data[i]);

        ++destination_iterator;
    }

    return destination;
}
