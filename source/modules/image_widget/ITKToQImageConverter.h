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
                          uint slice_index);
    static QImage convert_mask(MaskImage::Pointer image);

    static void setWindowFrom(ImageType::PixelType value);
    static void setWindowTo(ImageType::PixelType value);
private:
    static ImageType::PixelType* window_from;
    static ImageType::PixelType* window_to;
};

#endif // ITKToQImageConverter_H
