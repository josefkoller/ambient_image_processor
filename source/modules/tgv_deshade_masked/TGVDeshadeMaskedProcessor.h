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

#ifndef TGVDESHADEMASKEDPROCESSOR_H
#define TGVDESHADEMASKEDPROCESSOR_H

#include "ITKImage.h"

#include <functional>

class TGVDeshadeMaskedProcessor
{
public:
    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               ITKImage l)> IterationFinished;
    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               ITKImage u, ITKImage l, ITKImage r)> IterationFinishedThreeImages;
    typedef ITKImage::PixelType Pixel;
    typedef std::vector<uint> IndexVector;
    typedef const IndexVector& IndexVectorConstReference;
private:
    TGVDeshadeMaskedProcessor();

    template<typename Pixel>
    using IterationCallback = std::function<bool(uint iteration_index, uint iteration_count, Pixel* u,
    Pixel* v_x, Pixel* v_y, Pixel* v_z)>;

    template<typename Pixel>
    using IterationCallback2D = std::function<bool(uint iteration_index, uint iteration_count, Pixel* u,
    Pixel* v_x, Pixel* v_y)>;

    template<typename Pixel>
    using TGVAlgorithm = std::function<Pixel*(Pixel* f, IterationCallback<Pixel> iteration_callback,
    Pixel** v_x, Pixel**v_y, Pixel**v_z)>;
public:

    static ITKImage processTGV2L1GPUCuda(ITKImage input_image,
                                         const Pixel lambda,
                                         const Pixel alpha0,
                                         const Pixel alpha1,
                                         const uint iteration_count,
                                         const int cuda_block_dimension,
                                         const uint paint_iteration_interval,
                                         IterationFinishedThreeImages iteration_finished_callback,
                                         ITKImage mask,
                                         const bool set_negative_values_to_zero,
                                         const bool add_background_back,
                                         ITKImage& denoised_image,
                                         ITKImage& shading_image,
                                         ITKImage& div_v_image,
                                         const bool calculate_div_v);

    static ITKImage processTGV2L1GPUCuda2D(ITKImage input_image,
                                         const Pixel lambda,
                                         const Pixel alpha0,
                                         const Pixel alpha1,
                                         const uint iteration_count,
                                         const int cuda_block_dimension,
                                         const uint paint_iteration_interval,
                                         IterationFinishedThreeImages iteration_finished_callback,
                                         ITKImage mask,
                                         const bool set_negative_values_to_zero,
                                         const bool add_background_back,
                                         ITKImage& denoised_image,
                                         ITKImage& shading_image,
                                         ITKImage& div_v_image,
                                         const bool calculate_div_v);


    static void buildMaskIndices(ITKImage mask,
                                 IndexVector& pixel_indices_inside_mask,
                                 IndexVector& left_edge_pixel_indices, IndexVector& not_left_edge_pixel_indices,
                                 IndexVector& right_edge_pixel_indices, IndexVector& not_right_edge_pixel_indices,

                                 IndexVector& top_edge_pixel_indices, IndexVector& not_top_edge_pixel_indices,
                                 IndexVector& bottom_edge_pixel_indices, IndexVector& not_bottom_edge_pixel_indices,

                                 IndexVector& front_edge_pixel_indices, IndexVector& not_front_edge_pixel_indices,
                                 IndexVector& back_edge_pixel_indices, IndexVector& not_back_edge_pixel_indices);

    static void buildMaskIndices2D(ITKImage mask,
                                 IndexVector& pixel_indices_inside_mask,
                                 IndexVector& left_edge_pixel_indices, IndexVector& not_left_edge_pixel_indices,
                                 IndexVector& right_edge_pixel_indices, IndexVector& not_right_edge_pixel_indices,

                                 IndexVector& top_edge_pixel_indices, IndexVector& not_top_edge_pixel_indices,
                                 IndexVector& bottom_edge_pixel_indices, IndexVector& not_bottom_edge_pixel_indices);
};

#endif // TGVDESHADEMASKEDPROCESSOR_H
