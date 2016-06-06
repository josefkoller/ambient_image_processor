#ifndef TGVDESHADEPROCESSOR_H
#define TGVDESHADEPROCESSOR_H

#include "ITKImage.h"

#include <functional>

class TGVDeshadeProcessor
{
public:
    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               ITKImage l)> IterationFinished;
private:
    TGVDeshadeProcessor();

    typedef ITKImage::PixelType Pixel;

    template<typename Pixel>
    using IterationCallback = std::function<bool(uint iteration_index, uint iteration_count, Pixel* u,
        Pixel* v_x, Pixel* v_y, Pixel* v_z)>;

    template<typename Pixel>
    using TGVAlgorithm = std::function<Pixel*(Pixel* f, IterationCallback<Pixel> iteration_callback,
        Pixel** v_x, Pixel**v_y, Pixel**v_z)>;

    static ITKImage processTVGPUCuda(ITKImage input_image,
                                IterationFinished iteration_finished_callback,
                                TGVAlgorithm<Pixel> tgv_algorithm);
public:
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

    static ITKImage integrate_image_gradients(ITKImage gradient_x, ITKImage gradient_y, ITKImage gradient_z);
};

#endif // TGVDESHADEPROCESSOR_H
