#ifndef TGVDESHADEINTEGRALMETRICPLOTPROCESSOR_H
#define TGVDESHADEINTEGRALMETRICPLOTPROCESSOR_H

#include <vector>
#include "ITKImage.h"

#include "TGVDeshadeProcessor.h"
#include "cosine_transform.h"

class TGVDeshadeIntegralMetricPlotProcessor
{
public:
    typedef double ParameterValue;
    typedef std::pair<ParameterValue, ParameterValue> ParameterSet;
    typedef std::vector<ParameterSet> ParameterList;

    typedef ITKImage::PixelType Pixel;
    typedef ITKImage::PixelType MetricValue;

    enum MetricType
    {
        NormalizedCrossCorrelation = 0,
        SumOfAbsoluteDifferences,
        CoefficientOfVariationDeshaded,
        EntropyDeshaded,
    };

private:
    TGVDeshadeIntegralMetricPlotProcessor();

    template<typename Pixel>
    using IterationFinishedCuda = std::function<bool(uint iteration_index, uint iteration_count,
                               Pixel* metricValues,
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
            const ParameterList alpha_list,
            const uint iteration_count,
            const ITKImage& mask,
            const MetricType metric_type,
            const Pixel entropy_kernel_bandwidth,

            const uint paint_iteration_interval,
            IterationFinished iteration_callback,

            ITKImage& denoised_image,
            ITKImage& shading_image,
            ITKImage& deshaded_image);
};

#endif // TGVDESHADEINTEGRALMETRICPLOTPROCESSOR_H
