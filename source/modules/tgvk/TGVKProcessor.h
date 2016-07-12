#ifndef TGVKPROCESSOR_H
#define TGVKPROCESSOR_H

#include <ITKImage.h>
#include "TGV3Processor.h"

class TGVKProcessor
{
private:
    TGVKProcessor();
public:
    typedef TGV3Processor::IterationFinished IterationFinished;
    template<typename Pixel>
    using IterationCallback = TGV3Processor::IterationCallback<Pixel>;

    typedef ITKImage::PixelType Pixel;

    static ITKImage processTGVKL1GPUCuda(ITKImage input_image,
      const Pixel lambda,
      const uint order,
      const Pixel* alpha,
      const uint iteration_count,
      const uint paint_iteration_interval, IterationFinished iteration_finished_callback);
};

#endif // TGVKPROCESSOR_H
