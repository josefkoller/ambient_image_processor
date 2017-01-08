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

#ifndef TGVKDESHADEMASKEDPROCESSOR_H
#define TGVKDESHADEMASKEDPROCESSOR_H

#include <functional>
#include "ITKImage.h"

class TGVKDeshadeMaskedProcessor
{
private:
    TGVKDeshadeMaskedProcessor();

public:
    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               ITKImage l)> IterationFinished;
    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               ITKImage u, ITKImage l, ITKImage r)> IterationFinishedThreeImages;
    typedef ITKImage::PixelType Pixel;

    typedef std::vector<uint> IndexVector;
    typedef const IndexVector& IndexVectorConstReference;

    template<typename Pixel>
    using IterationCallback = std::function<bool(uint iteration_index, uint iteration_count, Pixel* u,
    Pixel* v_x, Pixel* v_y, Pixel* v_z)>;

    template<typename Pixel>
    using IterationCallback2D = std::function<bool(uint iteration_index, uint iteration_count, Pixel* u,
    Pixel* v_x, Pixel* v_y)>;

    static void processTGVKL1Cuda(ITKImage input_image,
                                 const Pixel lambda,

                                  const uint order,
                                  const Pixel* alpha,

                                 const uint iteration_count,
                                 const int cuda_block_dimension,
                                 ITKImage mask_image,
                                 const bool set_negative_values_to_zero,
                                 const bool add_background_back,

                                 const uint paint_iteration_interval,
                                 IterationFinishedThreeImages iteration_finished_callback,

                                 ITKImage& denoised_image,
                                 ITKImage& shading_image,
                                 ITKImage& deshaded_image,
                                 ITKImage& div_v_image,
                                 const bool calculate_div_v);

    static void processTGVKL1Cuda2D(ITKImage input_image,
                                 const Pixel lambda,

                                  const uint order,
                                  const Pixel* alpha,

                                 const uint iteration_count,
                                 const int cuda_block_dimension,
                                 ITKImage mask_image,
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

#endif // TGVKDESHADEMASKEDPROCESSOR_H
