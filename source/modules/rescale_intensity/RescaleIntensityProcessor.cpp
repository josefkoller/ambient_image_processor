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

#include "RescaleIntensityProcessor.h"

#include "CudaImageOperationsProcessor.h"

RescaleIntensityProcessor::RescaleIntensityProcessor()
{
}

ITKImage RescaleIntensityProcessor::process(ITKImage image, ITKImage::PixelType from, ITKImage::PixelType to)
{
    ITKImage::PixelType minimum, maximum;
    image.minimumAndMaximum(minimum, maximum);

    return process(image, from, to, minimum, maximum);
}

ITKImage RescaleIntensityProcessor::process(ITKImage image, ITKImage::PixelType from, ITKImage::PixelType to,
                                            ITKImage mask)
{
    if(mask.isNull())
        return process(image, from, to);

    ITKImage::PixelType minimum, maximum;
    image.minimumAndMaximumInsideMask(minimum, maximum, mask);

    return process(image, from, to, minimum, maximum);
}

ITKImage RescaleIntensityProcessor::process(ITKImage image, ITKImage::PixelType from, ITKImage::PixelType to,
                                            ITKImage::PixelType minimum, ITKImage::PixelType maximum)
{
    auto range_before = maximum - minimum;
    auto range_after = to - from;

    auto scale = range_after / range_before;

    image = CudaImageOperationsProcessor::addConstant(image, -minimum);
    image = CudaImageOperationsProcessor::multiplyConstant(image, scale);

    return CudaImageOperationsProcessor::addConstant(image, from);
}
