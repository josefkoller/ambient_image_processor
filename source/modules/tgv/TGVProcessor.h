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

#ifndef TGVPROCESSOR_H
#define TGVPROCESSOR_H

#include <ITKImage.h>

class TGVProcessor
{
private:
    TGVProcessor();

public:
    typedef ITKImage::PixelType Pixel;

    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               ITKImage u)> IterationFinished;

    template<typename Pixel>
    using IterationCallback = std::function<bool(uint iteration_index, uint iteration_count, Pixel* u)>;

private:

    template<typename Pixel>
    using TGVAlgorithm = std::function<Pixel*(Pixel* f, IterationCallback<Pixel> iteration_callback)>;
public:

    static ITKImage processTVGPUCuda(ITKImage input_image,
                                IterationFinished iteration_finished_callback,
                                TGVAlgorithm<Pixel> tgv_algorithm);

    static ITKImage processTVL2GPUCuda(ITKImage input_image,
      const Pixel lambda,
      const Pixel alpha0,
      const Pixel alpha1,
      const uint iteration_count,
      const uint paint_iteration_interval, IterationFinished iteration_finished_callback);

    static ITKImage processTVL1GPUCuda(ITKImage input_image,
      const Pixel lambda,
      const Pixel alpha0,
      const Pixel alpha1,
      const uint iteration_count,
      const uint paint_iteration_interval, IterationFinished iteration_finished_callback);

    static ITKImage processTGV2L1GPUCuda(ITKImage input_image,
      const Pixel lambda,
      const Pixel alpha0,
      const Pixel alpha1,
      const uint iteration_count,
      const uint paint_iteration_interval, IterationFinished iteration_finished_callback);

    static ITKImage processTGV2L2GPUCuda(ITKImage input_image,
      const Pixel lambda,
      const Pixel alpha0,
      const Pixel alpha1,
      const uint iteration_count,
      const uint paint_iteration_interval, IterationFinished iteration_finished_callback);
};

#endif // TGVPROCESSOR_H
