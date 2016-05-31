#ifndef TGVLAMBDASPROCESSOR_H
#define TGVLAMBDASPROCESSOR_H

#include "ITKImage.h"

class TGVLambdasProcessor
{
private:
    TGVLambdasProcessor();

    typedef ITKImage::PixelType Pixel;
public:
    typedef std::function<void(uint iteration_index, uint iteration_count,
                               ITKImage u)> IterationFinished;

    template<typename Pixel>
    using IterationCallback = std::function<void(uint iteration_index, uint iteration_count, Pixel* u)>;

    static ITKImage processTGV2L1LambdasGPUCuda(ITKImage input_image,
                                                ITKImage lambdas_image,
      const ITKImage::PixelType lambda_factor,
      const ITKImage::PixelType alpha0,
      const ITKImage::PixelType alpha1,
      const uint iteration_count,
      const uint paint_iteration_interval, IterationFinished iteration_finished_callback);

};

#endif // TGVLAMBDASPROCESSOR_H
