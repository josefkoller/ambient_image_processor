#ifndef TGV3PROCESSOR_H
#define TGV3PROCESSOR_H

#include <functional>

#include "ITKImage.h"

class TGV3Processor
{
private:
    TGV3Processor();
public:
    typedef ITKImage::PixelType Pixel;

    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               ITKImage u)> IterationFinished;

    template<typename Pixel>
    using IterationCallback = std::function<bool(uint iteration_index, uint iteration_count, Pixel* u)>;

    static ITKImage processTGV3L1GPUCuda(ITKImage input_image,
      const Pixel lambda,
      const Pixel alpha0,
      const Pixel alpha1,
      const Pixel alpha2,
      const uint iteration_count,
      const uint paint_iteration_interval, IterationFinished iteration_finished_callback);
};

#endif // TGV3PROCESSOR_H
