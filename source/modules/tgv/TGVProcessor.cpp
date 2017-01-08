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

#include "TGVProcessor.h"


#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIteratorWithIndex.h>

#include <functional>

template<typename Pixel>
Pixel* tgv1_l2_launch(Pixel* f_host,
                      uint width, uint height, uint depth,
                      Pixel lambda,
                      uint iteration_count,
                      uint paint_iteration_interval,
                      TGVProcessor::IterationCallback<Pixel> iteration_finished_callback,
                      Pixel alpha0,
                      Pixel alpha1);

template<typename Pixel>
Pixel* tgv1_l1_launch(Pixel* f_host,
                      uint width, uint height, uint depth,
                      Pixel lambda,
                      uint iteration_count,
                      uint paint_iteration_interval,
                      TGVProcessor::IterationCallback<Pixel> iteration_finished_callback,
                      Pixel alpha0,
                      Pixel alpha1);

template<typename Pixel>
Pixel* tgv2_l1_launch(Pixel* f_host,
                      uint width, uint height, uint depth,
                      Pixel lambda,
                      uint iteration_count,
                      uint paint_iteration_interval,
                      TGVProcessor::IterationCallback<Pixel> iteration_finished_callback,
                      Pixel alpha0,
                      Pixel alpha1);

template<typename Pixel>
Pixel* tgv2_l2_launch(Pixel* f_host,
                      uint width, uint height, uint depth,
                      Pixel lambda,
                      uint iteration_count,
                      uint paint_iteration_interval,
                      TGVProcessor::IterationCallback<Pixel> iteration_finished_callback,
                      Pixel alpha0,
                      Pixel alpha1);

TGVProcessor::TGVProcessor()
{
}

ITKImage TGVProcessor::processTVGPUCuda(ITKImage input_image,
                                        IterationFinished iteration_finished_callback,
                                        TGVAlgorithm<Pixel> tgv_algorithm)
{
    Pixel* f = input_image.cloneToPixelArray();

    IterationCallback<Pixel> iteration_callback = [&input_image, iteration_finished_callback] (
            uint iteration_index, uint iteration_count, Pixel* u) {
        auto itk_u = ITKImage(input_image.width, input_image.height, input_image.depth, u);
        return iteration_finished_callback(iteration_index, iteration_count, itk_u);
    };

    Pixel* u = tgv_algorithm(f, iteration_callback);


    delete[] f;

    auto result = ITKImage(input_image.width, input_image.height, input_image.depth, u);
    delete[] u;
    return result;
}


ITKImage TGVProcessor::processTVL2GPUCuda(ITKImage input_image,
                                          const Pixel lambda,
                                          const Pixel alpha0,
                                          const Pixel alpha1,
                                          const uint iteration_count,
                                          const uint paint_iteration_interval,
                                          IterationFinished iteration_finished_callback)
{
    return processTVGPUCuda(input_image, iteration_finished_callback,
        [&input_image, lambda, iteration_count, paint_iteration_interval, alpha0, alpha1]
        (Pixel* f, IterationCallback<Pixel> iteration_callback) {
        return tgv1_l2_launch<Pixel>(f,
                                     input_image.width, input_image.height, input_image.depth,
                                     lambda,
                                     iteration_count,
                                     paint_iteration_interval,
                                     iteration_callback,
                                     alpha0, alpha1);
    });
}

ITKImage TGVProcessor::processTVL1GPUCuda(ITKImage input_image,
                                          const Pixel lambda,
                                          const Pixel alpha0,
                                          const Pixel alpha1,
                                          const uint iteration_count,
                                          const uint paint_iteration_interval,
                                          IterationFinished iteration_finished_callback)
{
    return processTVGPUCuda(input_image, iteration_finished_callback,
        [&input_image, lambda, iteration_count, paint_iteration_interval, alpha0, alpha1]
        (Pixel* f, IterationCallback<Pixel> iteration_callback) {
        return tgv1_l1_launch<Pixel>(f,
                                     input_image.width, input_image.height, input_image.depth,
                                     lambda,
                                     iteration_count,
                                     paint_iteration_interval,
                                     iteration_callback,
                                     alpha0, alpha1);
    });
}


ITKImage TGVProcessor::processTGV2L1GPUCuda(ITKImage input_image,
                                          const Pixel lambda,
                                          const Pixel alpha0,
                                          const Pixel alpha1,
                                          const uint iteration_count,
                                          const uint paint_iteration_interval,
                                          IterationFinished iteration_finished_callback)
{
    return processTVGPUCuda(input_image, iteration_finished_callback,
        [&input_image, lambda, iteration_count, paint_iteration_interval, alpha0, alpha1]
        (Pixel* f, IterationCallback<Pixel> iteration_callback) {
        return tgv2_l1_launch<Pixel>(f,
                                     input_image.width, input_image.height, input_image.depth,
                                     lambda,
                                     iteration_count,
                                     paint_iteration_interval,
                                     iteration_callback,
                                     alpha0, alpha1);
    });
}

ITKImage TGVProcessor::processTGV2L2GPUCuda(ITKImage input_image,
                                          const Pixel lambda,
                                          const Pixel alpha0,
                                          const Pixel alpha1,
                                          const uint iteration_count,
                                          const uint paint_iteration_interval,
                                          IterationFinished iteration_finished_callback)
{
    return processTVGPUCuda(input_image, iteration_finished_callback,
        [&input_image, lambda, iteration_count, paint_iteration_interval, alpha0, alpha1]
        (Pixel* f, IterationCallback<Pixel> iteration_callback) {
        return tgv2_l2_launch<Pixel>(f,
                                     input_image.width, input_image.height, input_image.depth,
                                     lambda,
                                     iteration_count,
                                     paint_iteration_interval,
                                     iteration_callback,
                                     alpha0, alpha1);
    });
}
