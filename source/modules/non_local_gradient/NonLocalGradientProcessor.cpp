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
