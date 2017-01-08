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

#include "TGV3Processor.h"

template<typename Pixel>
Pixel* tgv3_l1_launch(Pixel* f_host,
                      uint width, uint height, uint depth,
                      Pixel lambda,
                      uint iteration_count,
                      uint paint_iteration_interval,
                      TGV3Processor::IterationCallback<Pixel> iteration_finished_callback,
                      Pixel alpha0,
                      Pixel alpha1,
                      Pixel alpha2);

TGV3Processor::TGV3Processor()
{
}


ITKImage TGV3Processor::processTGV3L1GPUCuda(ITKImage input_image,
  const Pixel lambda,
  const Pixel alpha0,
  const Pixel alpha1,
  const Pixel alpha2,
  const uint iteration_count,
  const uint paint_iteration_interval, IterationFinished iteration_finished_callback)
{
    Pixel* f = input_image.cloneToPixelArray();

    IterationCallback<Pixel> iteration_callback = [&input_image, iteration_finished_callback] (
            uint iteration_index, uint iteration_count, Pixel* u) {
        auto itk_u = ITKImage(input_image.width, input_image.height, input_image.depth, u);
        return iteration_finished_callback(iteration_index, iteration_count, itk_u);
    };

    Pixel* u = tgv3_l1_launch(f,
                              input_image.width, input_image.height, input_image.depth,
                              lambda,
                              iteration_count,
                              paint_iteration_interval,
                              iteration_callback,
                              alpha0, alpha1, alpha2);

    delete[] f;

    auto result = ITKImage(input_image.width, input_image.height, input_image.depth, u);
    delete[] u;
    return result;
}
