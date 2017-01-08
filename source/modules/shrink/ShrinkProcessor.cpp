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

#include "ShrinkProcessor.h"

#include <itkShrinkImageFilter.h>

ShrinkProcessor::ShrinkProcessor()
{
}


ITKImage ShrinkProcessor::process(ITKImage image,
            unsigned int shrink_factor_x,
            unsigned int shrink_factor_y,
            unsigned int shrink_factor_z)
{
    if(image.isNull())
        return ITKImage();

    typedef ITKImage::InnerITKImage Image;

    typedef itk::ShrinkImageFilter<Image, Image> Shrinker;
    typename Shrinker::Pointer shrinker = Shrinker::New();
    shrinker->SetInput( image.getPointer() );
    shrinker->SetShrinkFactor(0, shrink_factor_x);
    shrinker->SetShrinkFactor(1, shrink_factor_y);
    shrinker->SetShrinkFactor(2, shrink_factor_z);

    shrinker->Update();
    Image::Pointer result = shrinker->GetOutput();
    result->DisconnectPipeline();

    return ITKImage(result);
}
