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

#ifndef EXTRACTPROCESSOR_H
#define EXTRACTPROCESSOR_H

#include "ITKImage.h"

class ExtractProcessor
{
private:
    ExtractProcessor();
public:
    static ITKImage process(ITKImage image,
         unsigned int from_x, unsigned int to_x,
         unsigned int from_y, unsigned int to_y,
         unsigned int from_z, unsigned int to_z);

    static ITKImage process(ITKImage image, ITKImage::InnerITKImage::RegionType region);

    static ITKImage extract_slice(ITKImage image,
         unsigned int slice_index);
};

#endif // EXTRACTPROCESSOR_H
