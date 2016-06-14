#ifndef TGVDESHADEMETRICPLOTPROCESSOR_H
#define TGVDESHADEMETRICPLOTPROCESSOR_H

#include <vector>
#include "ITKImage.h"

#include "TGVDeshadeProcessor.h"
#include "cosine_transform.h"

class TGVDeshadeMetricPlotProcessor
{
public:
    typedef ITKImage::PixelType Pixel;
    typedef ITKImage::PixelType MetricValue;

private:
    TGVDeshadeMetricPlotProcessor();

    template<typename Pixel>
    using IterationFinishedCuda = std::function<bool(uint iteration_index, uint iteration_count,
                               MetricValue* metricValues,
                               Pixel* u, Pixel* l, Pixel* r)>;

    template<typename Pixel>
    using CosineTransformCallback = std::function<void(Pixel* image,
                               DimensionSize width, DimensionSize height, DimensionSize depth,
                               Pixel* result, bool is_inverse)>;
public:
    typedef std::vector<MetricValue> MetricValues;

    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               MetricValues metricValues,
                               ITKImage u, ITKImage l, ITKImage r)> IterationFinished;

    static MetricValues processTGV2L1DeshadeCuda(
            ITKImage input_image,
            const Pixel lambda,
            const Pixel alpha0,
            const Pixel alpha1,
            const uint iteration_count,
            const ITKImage& mask,

            const uint paint_iteration_interval,
            IterationFinished iteration_callback,

            ITKImage& denoised_image,
            ITKImage& shading_image,
            ITKImage& deshaded_image);
};

#endif // TGVDESHADEMETRICPLOTPROCESSOR_H
