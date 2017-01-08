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

#ifndef ITKToQImageConverter_H
#define ITKToQImageConverter_H

#include <QImage>
#include "ITKImage.h"

class ITKToQImageConverter
{
public:
    static QImage* convert(ITKImage image,
                           ITKImage mask,
                           uint slice_index,
                           bool do_rescale,
                           bool do_multiply,
                           bool use_window);

    static void setWindowFrom(ITKImage::PixelType value);
    static void setWindowTo(ITKImage::PixelType value);

    static const QColor lower_window_color;
    static const QColor upper_window_color;
    static const QColor outside_mask_color;
private:
    static ITKImage::PixelType* window_from;
    static ITKImage::PixelType* window_to;
};

#endif // ITKToQImageConverter_H
