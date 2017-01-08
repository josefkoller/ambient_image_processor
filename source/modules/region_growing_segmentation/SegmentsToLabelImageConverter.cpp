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

#include "SegmentsToLabelImageConverter.h"

SegmentsToLabelImageConverter::SegmentsToLabelImageConverter()
{

}


SegmentsToLabelImageConverter::LabelImage
    SegmentsToLabelImageConverter::convert(SegmentVector segments, LabelImage::InnerITKImage::SizeType size)
{
    LabelImage label_image = LabelImage(size[0], size[1], size[2]);
    label_image.setEachPixel([](uint,uint,uint) {
        return 0;
    });

    LabelImage::PixelType segment_pixel_value = 1;
    for(RegionGrowingSegmentation::Segment segment : segments)
    {
        for(auto seed_point : segment.seed_points)
        {
            label_image.setPixel(seed_point.position, segment_pixel_value);
        }
        segment_pixel_value++;
    }
    return label_image;
}
