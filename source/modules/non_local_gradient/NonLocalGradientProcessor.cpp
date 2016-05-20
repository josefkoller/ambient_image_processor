#include "NonLocalGradientProcessor.h"

#include <itkImageRegionIteratorWithIndex.h>

#include <iostream>

template<typename Pixel>
Pixel* non_local_gradient_kernel_launch(
        Pixel* source, uint source_width, uint source_height, uint source_depth,
        Pixel* kernel, uint kernel_size);

NonLocalGradientProcessor::NonLocalGradientProcessor()
{
}

ITKImage NonLocalGradientProcessor::createKernel(
        uint kernel_size,
        ITKImage::PixelType kernel_sigma)
{
    ITKImage kernel = ITKImage(kernel_size, kernel_size, kernel_size);
    uint kernel_center = std::floor(kernel_size / 2.0f);

    ITKImage::PixelType kernel_value_sum = 0;

    kernel.setEachPixel([&kernel_value_sum, kernel_center, kernel_sigma] (uint x, uint y, uint z) {
        uint xr = x - kernel_center;
        uint yr = y - kernel_center;
        uint zr = z - kernel_center;

        ITKImage::PixelType radius = std::sqrt(xr*xr + yr*yr + zr*zr);
        ITKImage::PixelType value = std::exp(-radius*radius / kernel_sigma);

        kernel_value_sum += value;
        return value;
    });

    kernel.foreachPixel([&kernel, kernel_value_sum] (uint x, uint y, uint z, ITKImage::PixelType value) {
        kernel.setPixel(x,y,z, value / kernel_value_sum);
    });

    return kernel;
}

ITKImage NonLocalGradientProcessor::process(ITKImage source,
                                            uint kernel_size,
                                            ITKImage::PixelType kernel_sigma)
{
    ITKImage::PixelType* source_data = source.cloneToPixelArray();
    ITKImage kernel = createKernel(kernel_size, kernel_sigma);
    ITKImage::PixelType* kernel_data = kernel.cloneToPixelArray();

    ITKImage::PixelType* destination_data = non_local_gradient_kernel_launch(source_data,
         source.width, source.height, source.depth, kernel_data, kernel_size);

    delete[] source_data;
    delete[] kernel_data;

    ITKImage destination = ITKImage(source.width, source.height, source.depth, destination_data);

    delete[] destination_data;

    return destination;
}
