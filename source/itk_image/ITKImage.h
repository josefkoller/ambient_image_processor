#ifndef ITKIMAGE_H
#define ITKIMAGE_H

#include <itkImage.h>


class ITKImage
{
public:
    typedef unsigned int uint;
    const uint ImageDimension = 2;
    typedef double PixelType;

    typedef itk::Image<PixelType, ImageDimension> InnerITKImage;
private:
    uint width;
    uint height;

    InnerITKImage::Pointer inner_image;
public:
    ITKImage(uint width, uint height);

    InnerITKImage::Pointer getPointer() const;
    InnerITKImage::Pointer clone() const;
};

#endif // ITKIMAGE_H
