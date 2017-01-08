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

#ifndef CUDAIMAGEOPERATIONSPROCESSOR_H
#define CUDAIMAGEOPERATIONSPROCESSOR_H

#include "ITKImage.h"
#include <functional>

class CudaImageOperationsProcessor
{
private:
    CudaImageOperationsProcessor();

    typedef ITKImage::PixelType Pixel;
    typedef Pixel* Pixels;
    typedef std::function<Pixels(Pixels, Pixels)> BinaryPixelsOperation;
    typedef std::function<Pixels(Pixels)> UnaryPixelsOperation;

    static ITKImage perform(ITKImage image1, ITKImage image2, BinaryPixelsOperation operation);
    static ITKImage perform(ITKImage image, UnaryPixelsOperation operation);
public:
    static ITKImage multiply(ITKImage image1, ITKImage image2);
    static ITKImage divide(ITKImage image1, ITKImage image2);
    static ITKImage add(ITKImage image1, ITKImage image2);
    static ITKImage subtract(ITKImage image1, ITKImage image2);

    static ITKImage addConstant(ITKImage image, ITKImage::PixelType constant);
    static ITKImage multiplyConstant(ITKImage image, ITKImage::PixelType constant);

    static ITKImage convolution3x3(ITKImage image, ITKImage::PixelType* kernel, bool calculate_center_as_sum_of_others);
    static ITKImage convolution3x3x3(ITKImage image, ITKImage::PixelType* kernel, bool calculate_center_as_sum_of_others);

    static ITKImage cosineTransform(ITKImage image);
    static ITKImage inverseCosineTransform(ITKImage image);

    static ITKImage log(ITKImage image);
    static ITKImage exp(ITKImage image);

    static Pixel* divergence(Pixel* dx, Pixel* dy, Pixel* dz,
                               const uint width, const uint height, const uint depth, bool is_host_data=false);

    static Pixel* divergence_2d(Pixel* dx, Pixel* dy,
                               const uint width, const uint height, bool is_host_data=false);

    static ITKImage solvePoissonInCosineDomain(ITKImage image);

    static ITKImage invert(ITKImage image);
    static ITKImage binary_dilate(ITKImage image);

    static ITKImage clamp_negative_values(ITKImage image, ITKImage::PixelType value);

    static double tv(ITKImage image);

    static ITKImage binarize(ITKImage image);
    static ITKImage divGrad(ITKImage image);
    static ITKImage remove_zero_frequency(ITKImage image);
    static ITKImage rotate180InPlane(ITKImage image);
};

#endif // CUDAIMAGEOPERATIONSPROCESSOR_H
