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
