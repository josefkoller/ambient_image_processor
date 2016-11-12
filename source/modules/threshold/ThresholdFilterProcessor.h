#ifndef THRESHOLDFILTERPROCESSOR_H
#define THRESHOLDFILTERPROCESSOR_H

#include "ITKImage.h"

class ThresholdFilterProcessor
{
private:
    ThresholdFilterProcessor();

public:
    typedef ITKImage::InnerITKImage ImageType;

    static ITKImage process(ITKImage image,
                            ImageType::PixelType lower_threshold_value,
                            ImageType::PixelType upper_threshold_value,
                            ImageType::PixelType outside_pixel_value);

    static ITKImage clamp(ITKImage image,
                          ImageType::PixelType lower_threshold_value,
                          ImageType::PixelType upper_threshold_value);

};

#endif // THRESHOLDFILTERPROCESSOR_H
