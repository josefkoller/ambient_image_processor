#include "TGVL1ThresholdGradientProcessor.h"

template<typename Pixel>
Pixel* gradient_magnitude_kernel_launch(Pixel* f_host,
                                        uint width, uint height, uint depth);

template<typename Pixel>
Pixel* threshold_upper_kernel_launch(Pixel* f_host,
                                     uint width, uint height, uint depth,
                                     Pixel threshold_value);

template<typename Pixel>
void gradient_kernel_launch(Pixel* f_host,
                              uint width, uint height, uint depth,
                              Pixel** gradient_x_host,
                              Pixel** gradient_y_host,
                              Pixel** gradient_z_host);

#include <functional>

template<typename Pixel>
using IterationCallback = std::function<void(uint iteration_index, uint iteration_count, Pixel* u)>;


template<typename Pixel>
Pixel* tgv2_l1_threshold_gradient_launch(Pixel* f_host,
                                         Pixel* f_p_x_host,
                                         Pixel* f_p_y_host,
                                         Pixel* f_p_z_host,
                                         uint width, uint height, uint depth,
                                         Pixel lambda,
                                         uint iteration_count,
                                         uint paint_iteration_interval,
                                         IterationCallback<Pixel> iteration_finished_callback,
                                         Pixel alpha0,
                                         Pixel alpha1);

TGVL1ThresholdGradientProcessor::TGVL1ThresholdGradientProcessor()
{

}


ITKImage TGVL1ThresholdGradientProcessor::gradient_magnitude(ITKImage source_image) {
    auto source_pixels = source_image.cloneToPixelArray();

    auto result_pixels = gradient_magnitude_kernel_launch(source_pixels,
                                                          source_image.width,
                                                          source_image.height,
                                                          source_image.depth);
    auto result = ITKImage(source_image.width, source_image.height, source_image.depth, result_pixels);

    delete[] source_pixels;
    delete[] result_pixels;

    return result;
}

ITKImage TGVL1ThresholdGradientProcessor::threshold_upper_to_zero(
        ITKImage source_image, double threshold_value)
{
    auto source_pixels = source_image.cloneToPixelArray();

    auto result_pixels = threshold_upper_kernel_launch(source_pixels,
                                                       source_image.width,
                                                       source_image.height,
                                                       source_image.depth, threshold_value);
    auto result = ITKImage(source_image.width, source_image.height, source_image.depth, result_pixels);

    delete[] source_pixels;
    delete[] result_pixels;

    return result;
}


void TGVL1ThresholdGradientProcessor::gradient(ITKImage source_image,
                                               ITKImage& gradient_x,
                                               ITKImage& gradient_y,
                                               ITKImage& gradient_z) {
    auto source_pixels = source_image.cloneToPixelArray();
    ITKImage::PixelType* gradient_x_pixels = nullptr;
    ITKImage::PixelType* gradient_y_pixels = nullptr;
    ITKImage::PixelType* gradient_z_pixels = nullptr;

    gradient_kernel_launch(source_pixels,
                           source_image.width,
                           source_image.height,
                           source_image.depth,
                           &gradient_x_pixels,
                           &gradient_y_pixels,
                           &gradient_z_pixels);

    gradient_x = ITKImage(source_image.width, source_image.height, source_image.depth, gradient_x_pixels);
    gradient_y = ITKImage(source_image.width, source_image.height, source_image.depth, gradient_y_pixels);

    if(source_image.depth > 1)
        gradient_z = ITKImage(source_image.width, source_image.height, source_image.depth, gradient_z_pixels);

    delete[] source_pixels;
    delete[] gradient_x_pixels;
    delete[] gradient_y_pixels;

    if(source_image.depth > 1)
        delete[] gradient_z_pixels;
}

ITKImage TGVL1ThresholdGradientProcessor::tgv2_l1_threshold_gradient(ITKImage f_host,
                                         ITKImage f_p_x_host,
                                         ITKImage f_p_y_host,
                                         ITKImage f_p_z_host,
                                         ITKImage::PixelType lambda,
                                         uint iteration_count,
                                         uint paint_iteration_interval,
                                         IterationFinished iteration_finished_callback,
                                         ITKImage::PixelType alpha0,
                                         ITKImage::PixelType alpha1) {
    auto f = f_host.cloneToPixelArray();
    auto f_p_x = f_p_x_host.cloneToPixelArray();
    auto f_p_y = f_p_y_host.cloneToPixelArray();
    auto f_p_z = f_p_z_host.cloneToPixelArray();

    IterationCallback<ITKImage::PixelType> iteration_callback = [&f_host, iteration_finished_callback] (
            uint iteration_index, uint iteration_count, ITKImage::PixelType* u) {
        auto itk_u = ITKImage(f_host.width, f_host.height, f_host.depth, u);
        iteration_finished_callback(iteration_index, iteration_count, itk_u);
    };

    auto result_pixels = tgv2_l1_threshold_gradient_launch(f,
                                             f_p_x,
                                             f_p_y,
                                             f_p_z,
                                             f_host.width, f_host.height, f_host.depth,
                                             lambda,
                                             iteration_count,
                                             paint_iteration_interval,
                                             iteration_callback,
                                             alpha0,
                                             alpha1);
    auto result = ITKImage(f_host.width, f_host.height, f_host.depth, result_pixels);

    delete[] f;
    delete[] f_p_x;
    delete[] f_p_y;
    delete[] f_p_z;

    return result;
}
