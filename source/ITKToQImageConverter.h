#ifndef ITKToQImageConverter_H
#define ITKToQImageConverter_H

#include <QImage>
#include <QColor>

#include <itkRescaleIntensityImageFilter.h>

#include "ITKImageProcessor.h"

class ITKToQImageConverter
{
public:
    typedef ITKImageProcessor::ImageType ImageType;
    typedef ITKImageProcessor::MaskImage MaskImage;

    static QImage* convert(ImageType::Pointer image,
                          uint slice_index,
                          ImageType::PixelType window_from,
                          ImageType::PixelType window_to);
    static QImage convert_mask(MaskImage::Pointer image);
};

#endif // ITKToQImageConverter_H
