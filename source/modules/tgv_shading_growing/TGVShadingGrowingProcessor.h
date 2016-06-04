#ifndef TGVSHADINGGROWINGPROCESSOR_H
#define TGVSHADINGGROWINGPROCESSOR_H

#include "ITKImage.h"
#include "TGVProcessor.h"

class TGVShadingGrowingProcessor
{
private:
    TGVShadingGrowingProcessor();

    template<typename Pixel>
    using IterationCallback = TGVProcessor::IterationCallback<Pixel>;
public:
    typedef TGVProcessor::IterationFinished IterationFinished;

    static ITKImage process(ITKImage input_image,
                            const ITKImage::PixelType lambda,
                            const ITKImage::PixelType alpha0,
                            const ITKImage::PixelType alpha1,
                            const uint iteration_count,
                            const uint paint_iteration_interval, IterationFinished iteration_finished_callback,
                            ITKImage::PixelType lower_threshold,
                            ITKImage::PixelType upper_threshold,
                            ITKImage::PixelType non_local_gradient_kernel_sigma,
                            uint non_local_gradient_kernel_size);
};

#endif // TGVSHADINGGROWINGPROCESSOR_H
