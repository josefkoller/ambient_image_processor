#ifndef ITKRawImageConverter_H
#define ITKRawImageConverter_H

#include "../ITKImageProcessor.h"

#include "RawImage.h"

class ITKRawImageConverter
{
private:
    ITKRawImageConverter();
public:
    typedef ITKImageProcessor::ImageType ITKImage;

    static RawImage* convert(ITKImage::Pointer itk_image);
    static ITKImage::Pointer convert(RawImage* image);
    static ITKImage::SizeType convert(RawImage::Size size);
    static RawImage::Size convert(ITKImage::SizeType size);
};

#endif // ITKRawImageConverter_H
