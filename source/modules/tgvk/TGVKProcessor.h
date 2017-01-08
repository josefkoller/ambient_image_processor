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

#ifndef TGVKPROCESSOR_H
#define TGVKPROCESSOR_H

#include <ITKImage.h>
#include "TGV3Processor.h"

class TGVKProcessor
{
private:
    TGVKProcessor();
public:
    typedef TGV3Processor::IterationFinished IterationFinished;
    template<typename Pixel>
    using IterationCallback = TGV3Processor::IterationCallback<Pixel>;

    typedef ITKImage::PixelType Pixel;

    static ITKImage processTGVKL1GPUCuda(ITKImage input_image,
      const Pixel lambda,
      const uint order,
      const Pixel* alpha,
      const uint iteration_count,
      const uint paint_iteration_interval, IterationFinished iteration_finished_callback);
};

#endif // TGVKPROCESSOR_H
