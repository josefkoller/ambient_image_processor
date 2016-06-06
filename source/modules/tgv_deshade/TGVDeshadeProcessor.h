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

    static ITKImage deshade(Pixel* u, Pixel* v_x, Pixel* v_y, Pixel* v_z,
                            const uint width,
                            const uint height,
                            const uint depth);
    static ITKImage deshade_poisson_cosine_transform(Pixel* u, Pixel* v_x, Pixel* v_y, Pixel* v_z,
                             const uint width,
                             const uint height,
                             const uint depth, bool is_host_data=false);
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


    static ITKImage integrate_image_gradients_poisson_cosine_transform(Pixel* gradient_x,
                                                                                     Pixel* gradient_y,
                                                                                     Pixel* gradient_z,
                                                                                     const uint width,
                                                                                     const uint height,
                                                                                     const uint depth, bool is_host_data=false);
};

#endif // TGVDESHADEPROCESSOR_H
