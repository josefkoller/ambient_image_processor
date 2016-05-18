#ifndef ITKIMAGE_H
#define ITKIMAGE_H

#include <itkImage.h>
#include <string>

class ITKImage
{
public:
    typedef unsigned int uint;
    static const uint ImageDimension = 2;
    typedef double PixelType;

    typedef itk::Image<PixelType, ImageDimension> InnerITKImage;
private:
    uint width;
    uint height;

    InnerITKImage::Pointer inner_image;
public:
    ITKImage(uint width, uint height);
    ITKImage(InnerITKImage::Pointer inner_image);
    ITKImage(uint width, uint height, InnerITKImage::PixelType* data);

    InnerITKImage::Pointer getPointer() const;
    InnerITKImage::Pointer clone() const;

    static ITKImage read(std::string image_file_path);
    void write(std::string image_file_path);

    bool isNull() const;
};

#endif // ITKIMAGE_H
