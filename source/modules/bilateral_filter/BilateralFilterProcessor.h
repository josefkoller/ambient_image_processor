#ifndef BILATERALFILTERPROCESSOR_H
#define BILATERALFILTERPROCESSOR_H

#include "ITKImage.h"

class BilateralFilterProcessor
{
private:
    BilateralFilterProcessor();

    typedef ITKImage::InnerITKImage ImageType;
public:

    static ITKImage process(ITKImage image,
                            ImageType::PixelType sigma_spatial_distance,
                            ImageType::PixelType sigma_intensity_distance,
                            int kernel_size);

};

#endif // BILATERALFILTERPROCESSOR_H
