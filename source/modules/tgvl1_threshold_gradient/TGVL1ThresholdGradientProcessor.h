#ifndef TGVL1THRESHOLDGRADIENTPROCESSOR_H
#define TGVL1THRESHOLDGRADIENTPROCESSOR_H

#include "ITKImage.h"

class TGVL1ThresholdGradientProcessor
{
public:
    TGVL1ThresholdGradientProcessor();

    typedef std::function<void(uint iteration_index, uint iteration_count,
                               ITKImage u)> IterationFinished;

    static ITKImage gradient_magnitude(ITKImage source_image);

    static ITKImage threshold_upper_to_zero(
            ITKImage gradient_image, double threshold_value);


    static void gradient(ITKImage source_image,
                         ITKImage& gradient_x,
                         ITKImage& gradient_y,
                         ITKImage& gradient_z);

    static ITKImage tgv2_l1_threshold_gradient(ITKImage f_host,
                                             ITKImage f_p_x_host,
                                             ITKImage f_p_y_host,
                                             ITKImage f_p_z_host,
                                             ITKImage::PixelType lambda,
                                             uint iteration_count,
                                             uint paint_iteration_interval,
                                             IterationFinished iteration_finished_callback,
                                             ITKImage::PixelType alpha0,
                                             ITKImage::PixelType alpha1);
};

#endif // TGVL1THRESHOLDGRADIENTPROCESSOR_H
