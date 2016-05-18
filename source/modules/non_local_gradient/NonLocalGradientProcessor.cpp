#include "NonLocalGradientProcessor.h"

#include <itkImageRegionIteratorWithIndex.h>

#include <iostream>

template<typename Pixel>
Pixel* non_local_gradient_kernel_launch(
        Pixel* source, uint source_width, uint source_height, Pixel* kernel, uint kernel_size);

NonLocalGradientProcessor::NonLocalGradientProcessor()
{
}

ITKImage::PixelType* NonLocalGradientProcessor::createKernel(
        uint kernel_size,
        ITKImage::PixelType kernel_sigma)
{
    ITKImage::PixelType* kernel = new ITKImage::PixelType[kernel_size * kernel_size];
    uint kernel_center = std::floor(kernel_size / 2.0f);

    ITKImage::PixelType kernel_value_sum = 0;
    for(uint y = 0; y < kernel_size; y++)
    {
        for(uint x = 0; x < kernel_size; x++)
        {
            uint xr = x - kernel_center;
            uint yr = y - kernel_center;

            ITKImage::PixelType radius = std::sqrt(xr*xr + yr*yr);
            ITKImage::PixelType value = std::exp(-radius*radius / kernel_sigma);

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
            ITKImage::PixelType value = kernel[i];
            value /= kernel_value_sum;

            //            std::cout << "kernel value1: " << value << std::endl;

            kernel[i] = value;
        }
    }

    return kernel;
}

ITKImage NonLocalGradientProcessor::process(ITKImage source,
                                                       uint kernel_size,
                                                       ITKImage::PixelType kernel_sigma)
{
    ITKImage::PixelType* source_data = source.getPointer()->GetBufferPointer();
    ITKImage::PixelType* kernel_data = createKernel(kernel_size, kernel_sigma);

    ITKImage::PixelType* destination_data = non_local_gradient_kernel_launch(source_data,
                 source.width, source.height, kernel_data, kernel_size);

    delete[] kernel_data;

    ITKImage destination = ITKImage(source.width, source.height, destination_data);

    return destination;
}
