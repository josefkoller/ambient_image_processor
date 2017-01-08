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

#ifndef TGVLAMBDASPROCESSOR_H
#define TGVLAMBDASPROCESSOR_H

#include "ITKImage.h"

class TGVLambdasProcessor
{
private:
    TGVLambdasProcessor();

    typedef ITKImage::PixelType Pixel;
public:
    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               ITKImage u)> IterationFinished;

    template<typename Pixel>
    using IterationCallback = std::function<bool(uint iteration_index, uint iteration_count, Pixel* u)>;

    static ITKImage processTGV2L1LambdasGPUCuda(ITKImage input_image,
                                                ITKImage lambdas_image,
                                                const ITKImage::PixelType lambda_offset,
      const ITKImage::PixelType lambda_factor,
      const ITKImage::PixelType alpha0,
      const ITKImage::PixelType alpha1,
      const uint iteration_count,
      const uint paint_iteration_interval, IterationFinished iteration_finished_callback);

};

#endif // TGVLAMBDASPROCESSOR_H
