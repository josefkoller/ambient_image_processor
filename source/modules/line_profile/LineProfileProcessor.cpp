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

#include "LineProfileProcessor.h"

#include <itkLineConstIterator.h>
LineProfileProcessor::LineProfileProcessor()
{
}


void LineProfileProcessor::intensity_profile(ITKImage image,
                                             LineProfile::Point point1,
                                             LineProfile::Point point2,
                                             std::vector<double>& intensities,
                                             std::vector<double>& distances)
{
    if(!image.contains(point1) ||
       !image.contains(point2))
        return;

    typedef ITKImage::InnerITKImage ImageType;
    typedef ITKImage::PixelType PixelType;

    itk::LineConstIterator<ImageType> iterator(image.getPointer(), point1, point2);
    while(! iterator.IsAtEnd())
    {
        intensities.push_back(iterator.Get());

        ImageType::IndexType index = iterator.GetIndex();
        const int dx = point1[0] - index[0];
        const int dy = point1[1] - index[1];

        if(ImageType::ImageDimension == 3)
        {
            const int dz = point1[2] - index[2];
            distances.push_back(sqrt(dx*dx + dy*dy + dz*dz));
        }
        else
            distances.push_back(sqrt(dx*dx + dy*dy));

        ++iterator;
    }
}
