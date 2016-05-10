#ifndef ITKCIRCLEIMAGE_H
#define ITKCIRCLEIMAGE_H

#include "../CircleImage.h"

#include "../../ITKImageProcessor.h"
typedef ITKImageProcessor::ImageType ITKImage;

class ITKCircleImage
{
public:
    static ITKImage::Pointer create(CircleImage circle_image);
};

#endif // ITKCIRCLEIMAGE_H
