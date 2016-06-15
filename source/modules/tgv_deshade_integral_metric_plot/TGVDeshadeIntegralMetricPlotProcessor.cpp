#include "TGVDeshadeIntegralMetricPlotProcessor.h"

template<typename Pixel>
Pixel* tgv2_l1_deshade_integral_metrics_launch(Pixel* f_host,
      const uint width, const uint height, const uint depth,
      const Pixel lambda,
      const TGVDeshadeIntegralMetricPlotProcessor::ParameterList alpha_list,
      const uint iteration_count,
      Pixel* mask,
      const uint metric_type,
      const Pixel entropy_kernel_bandwidth,

      const uint paint_iteration_interval,
      TGVDeshadeIntegralMetricPlotProcessor::IterationFinishedCuda<Pixel> iteration_finished_callback,
      TGVDeshadeIntegralMetricPlotProcessor::CosineTransformCallback<Pixel> cosine_transform_callback,

      Pixel** denoised_pixels,
      Pixel** shading_pixels,
      Pixel** deshaded_pixels);


TGVDeshadeIntegralMetricPlotProcessor::TGVDeshadeIntegralMetricPlotProcessor()
{
}


TGVDeshadeIntegralMetricPlotProcessor::MetricValues TGVDeshadeIntegralMetricPlotProcessor::processTGV2L1DeshadeCuda(
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
        ITKImage& deshaded_image)
{
    auto input_pixels = input_image.cloneToPixelArray();
    auto mask_pixels = mask.cloneToPixelArray();

    IterationFinishedCuda<Pixel> iteration_callback_cuda = [&input_image, iteration_callback, &alpha_list]
            (uint iteration_index, uint iteration_count,
            MetricValue* metricValues,
            Pixel* u, Pixel* l, Pixel* r) {
        auto denoised_image = ITKImage(input_image.width, input_image.height, input_image.depth, u);
        auto shading_image = ITKImage(input_image.width, input_image.height, input_image.depth, l);
        auto deshaded_image = ITKImage(input_image.width, input_image.height, input_image.depth, r);

        auto metricValuesVector = MetricValues();
        for(int i = 0; i < alpha_list.size(); i++)
            metricValuesVector.push_back(metricValues[i]);

        return iteration_callback(iteration_index, iteration_count,
                                  metricValuesVector,
                                  denoised_image,
                                  shading_image,
                                  deshaded_image);
    };

    CosineTransformCallback<Pixel> cosine_transform_callback = [](Pixel* image,
            DimensionSize width, DimensionSize height, DimensionSize depth,
            Pixel* result, bool is_inverse = false) {
        if(is_inverse)
            inverse_cosine_transform(image, width, height, depth, result);
        else
            cosine_transform(image, width, height, depth, result);
    };

    Pixel* denoised_pixels, *shading_pixel, *deshaded_pixels;
    auto metricValues = tgv2_l1_deshade_integral_metrics_launch(
                input_pixels, input_image.width, input_image.height, input_image.depth,
                lambda, alpha_list, iteration_count, mask_pixels,
                (uint)metric_type,
                entropy_kernel_bandwidth,
                paint_iteration_interval,
                iteration_callback_cuda,
                cosine_transform_callback,
                &denoised_pixels, &shading_pixel, &deshaded_pixels);

    denoised_image = ITKImage(input_image.width, input_image.height, input_image.depth, denoised_pixels);
    shading_image = ITKImage(input_image.width, input_image.height, input_image.depth, shading_pixel);
    deshaded_image = ITKImage(input_image.width, input_image.height, input_image.depth, deshaded_pixels);

    auto metricValuesVector = MetricValues();
    for(int i = 0; i < alpha_list.size(); i++)
        metricValuesVector.push_back(metricValues[i]);

    delete[] input_pixels;
    delete[] mask_pixels;

    delete[] denoised_pixels;
    delete[] shading_pixel;
    delete[] metricValues;

    return metricValuesVector;
}
