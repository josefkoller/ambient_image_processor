#ifndef TGVKDESHADEPROCESSOR_H
#define TGVKDESHADEPROCESSOR_H

#include <functional>
#include "ITKImage.h"

class TGVKDeshadeProcessor
{
private:
    TGVKDeshadeProcessor();

public:
    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               ITKImage l)> IterationFinished;
    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               ITKImage u, ITKImage l, ITKImage r)> IterationFinishedThreeImages;
    typedef ITKImage::PixelType Pixel;

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


    static void processTGVKL1Cuda2D(ITKImage input_image,
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

#endif // TGVKDESHADEPROCESSOR_H
