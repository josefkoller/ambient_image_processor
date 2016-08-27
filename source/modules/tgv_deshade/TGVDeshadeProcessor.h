#ifndef TGVDESHADEPROCESSOR_H
#define TGVDESHADEPROCESSOR_H

#include "ITKImage.h"

#include <functional>

class TGVDeshadeProcessor
{
public:
    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               ITKImage l)> IterationFinished;
    typedef std::function<bool(uint iteration_index, uint iteration_count,
                               ITKImage u, ITKImage l, ITKImage r)> IterationFinishedThreeImages;
    typedef ITKImage::PixelType Pixel;
private:
    TGVDeshadeProcessor();


    template<typename Pixel>
    using IterationCallback = std::function<bool(uint iteration_index, uint iteration_count, Pixel* u,
    Pixel* v_x, Pixel* v_y, Pixel* v_z)>;

    template<typename Pixel>
    using IterationCallback2D = std::function<bool(uint iteration_index, uint iteration_count, Pixel* u,
    Pixel* v_x, Pixel* v_y)>;

    template<typename Pixel>
    using TGVAlgorithm = std::function<Pixel*(Pixel* f, IterationCallback<Pixel> iteration_callback,
    Pixel** v_x, Pixel**v_y, Pixel**v_z)>;

    static ITKImage processTVGPUCuda(ITKImage input_image,
                                     const ITKImage& mask,
                                     const bool set_negative_values_to_zero,
                                     const bool add_background_back,
                                     IterationFinishedThreeImages iteration_finished_callback,
                                     ITKImage& denoised_image,
                                     ITKImage& shading_image,
                                     TGVAlgorithm<Pixel> tgv_algorithm);

    static ITKImage deshade(Pixel* u, Pixel* v_x, Pixel* v_y, Pixel* v_z,
                            const uint width,
                            const uint height,
                            const uint depth);

    static ITKImage::PixelType time_tv(ITKImage image, ITKImage image_before);
    static ITKImage::PixelType mean(const ITKImage& image);
    static ITKImage::PixelType standard_deviation(const ITKImage& image, const ITKImage::PixelType mean);
    static ITKImage::PixelType normalized_cross_correlation(const ITKImage& image1, const ITKImage& image2);
public:

    static ITKImage deshade_poisson_cosine_transform(ITKImage u, Pixel* v_x, Pixel* v_y, Pixel* v_z,
                                                     const uint width,
                                                     const uint height,
                                                     const uint depth,
                                                     const ITKImage& mask,
                                                     const bool set_negative_values_to_zero,
                                                     ITKImage& l,
                                                     bool is_host_data=false);

    static ITKImage deshade_poisson_cosine_transform_2d(ITKImage u, Pixel* v_x, Pixel* v_y,
                                                     const uint width,
                                                     const uint height,
                                                     const ITKImage& mask,
                                                     const bool set_negative_values_to_zero,
                                                     ITKImage& l,
                                                     bool is_host_data=false);

    static ITKImage processTGV2L1GPUCuda(ITKImage input_image,
                                         const Pixel lambda,
                                         const Pixel alpha0,
                                         const Pixel alpha1,
                                         const uint iteration_count,
                                         const uint paint_iteration_interval,
                                         IterationFinishedThreeImages iteration_finished_callback,
                                         const ITKImage& mask,
                                         const bool set_negative_values_to_zero,
                                         const bool add_background_back,
                                         ITKImage& denoised_image,
                                         ITKImage& shading_image);

    static ITKImage processTGV2L1GPUCuda2D(ITKImage input_image,
                                         const Pixel lambda,
                                         const Pixel alpha0,
                                         const Pixel alpha1,
                                         const uint iteration_count,
                                         const uint paint_iteration_interval,
                                         IterationFinishedThreeImages iteration_finished_callback,
                                         const ITKImage& mask,
                                         const bool set_negative_values_to_zero,
                                         const bool add_background_back,
                                         ITKImage& denoised_image,
                                         ITKImage& shading_image);

    static ITKImage integrate_image_gradients(ITKImage gradient_x, ITKImage gradient_y, ITKImage gradient_z);


    static ITKImage integrate_image_gradients_poisson_cosine_transform(Pixel* gradient_x,
                                                                       Pixel* gradient_y,
                                                                       Pixel* gradient_z,
                                                                       const uint width,
                                                                       const uint height,
                                                                       const uint depth,
                                                                       bool is_host_data=false);

    static ITKImage integrate_image_gradients_poisson_cosine_transform_2d(Pixel* gradient_x,
                                                                       Pixel* gradient_y,
                                                                       const uint width,
                                                                       const uint height,
                                                                       bool is_host_data=false);

    static void processTGV2L1GPUCuda(ITKImage input_image,
                                 const Pixel lambda,
                                 const Pixel alpha0,
                                 const Pixel alpha1,
                                 const uint iteration_count,
                                 const ITKImage& mask_image,
                                 const bool set_negative_values_to_zero,
                                 ITKImage& denoised_image,
                                 ITKImage& shading_image,
                                 ITKImage& deshaded_image);

    typedef ITKImage::PixelType MetricValue;
    typedef std::vector<MetricValue> MetricValues;
    typedef std::vector<MetricValues> MetricValuesHistory;

    static MetricValuesHistory processTGV2L1DeshadeCuda_convergenceTest(
            ITKImage input_image,
            const Pixel lambda,
            const Pixel alpha0,
            const Pixel alpha1,
            const uint iteration_count,
            const uint check_iteration_interval,
            const ITKImage& mask,
            const bool set_negative_values_to_zero);
    static MetricValues processTGV2L1DeshadeCuda_convergenceTestMetric(
            ITKImage input_image,
            const Pixel lambda,
            const Pixel alpha0,
            const Pixel alpha1,
            const uint iteration_count,
            const uint check_iteration_interval,
            const ITKImage& mask,
            const bool set_negative_values_to_zero);
    static void processTGV2L1DeshadeCuda_convergenceTestToFile(
            ITKImage input_image,
            const Pixel lambda,
            const ITKImage& mask,
            const bool set_negative_values_to_zero,
            const Pixel alpha0,
            const Pixel alpha1,
            const uint check_iteration_count,
            std::string metric_file_name);
    static void processTGV2L1DeshadeCuda_convergenceOptimization(
            ITKImage input_image,
            const Pixel lambda,
            const ITKImage& mask,
            const bool set_negative_values_to_zero,
            ITKImage& denoised_image,
            ITKImage& shading_image,
            ITKImage& deshaded_image);
};

#endif // TGVDESHADEPROCESSOR_H
