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

#ifndef TGVKDeshadeDownsampledNOTMASKEDPROCESSOR_H
#define TGVKDeshadeDownsampledNOTMASKEDPROCESSOR_H

#include <functional>
#include "ITKImage.h"

class TGVKDeshadeDownsampledNotMaskedProcessor
{
private:
    TGVKDeshadeDownsampledNotMaskedProcessor();

public:
    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               ITKImage l)> IterationFinished;
    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               ITKImage u, ITKImage l, ITKImage r)> IterationFinishedThreeImages;
    typedef ITKImage::PixelType Pixel;

    template<typename Pixel>
    using IterationCallback = std::function<bool(uint iteration_index, uint iteration_count, Pixel* u,
    Pixel* v_x, Pixel* v_y, Pixel* v_z)>;

    static void processTGVKL1Cuda(ITKImage input_image,
                                 const Pixel downsampling_factor,
                                 const Pixel lambda,

                                  const uint order,
                                  const Pixel* alpha,

                                 const uint iteration_count,
                                 const ITKImage& mask_image,
                                 const bool set_negative_values_to_zero,
                                 const bool add_background_back,

                                 const uint paint_iteration_interval,
                                 IterationFinishedThreeImages iteration_finished_callback,

                                 ITKImage& denoised_image,
                                 ITKImage& shading_image,
                                 ITKImage& deshaded_image,
                                 ITKImage& div_v_image,
                                 const bool calculate_div_v);
};

#endif // TGVKDeshadeDownsampledNOTMASKEDPROCESSOR_H
