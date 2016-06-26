#ifndef TGV3DESHADEPROCESSOR_H
#define TGV3DESHADEPROCESSOR_H

#include <functional>
#include "ITKImage.h"

class TGV3DeshadeProcessor
{
private:
    TGV3DeshadeProcessor();

public:
    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               ITKImage l)> IterationFinished;
    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               ITKImage u, ITKImage l, ITKImage r)> IterationFinishedThreeImages;
    typedef ITKImage::PixelType Pixel;

    template<typename Pixel>
    using IterationCallback = std::function<bool(uint iteration_index, uint iteration_count, Pixel* u,
    Pixel* v_x, Pixel* v_y, Pixel* v_z)>;

    static void processTGV3L1Cuda(ITKImage input_image,
                                 const Pixel lambda,
                                 const Pixel alpha0,
                                 const Pixel alpha1,
                                 const Pixel alpha2,
                                 const uint iteration_count,
                                 const ITKImage& mask_image,
                                 const bool set_negative_values_to_zero,
                                 const bool add_background_back,

                                 const uint paint_iteration_interval,
                                 IterationFinishedThreeImages iteration_finished_callback,

                                 ITKImage& denoised_image,
                                 ITKImage& shading_image,
                                 ITKImage& deshaded_image);
};

#endif // TGV3DESHADEPROCESSOR_H
