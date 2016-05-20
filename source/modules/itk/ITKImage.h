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

    uint width;
    uint height;
    uint visible_slice_index;
private:
    InnerITKImage::Pointer inner_image;
public:
    ITKImage();
    ITKImage(const ITKImage&);
    ITKImage& operator=(ITKImage image)
    {
        this->width = image.width;
        this->height = image.height;
        this->inner_image = image.inner_image;
        return *this;
    }

    ITKImage(uint width, uint height);
    ITKImage(InnerITKImage::Pointer inner_image);
    ITKImage(uint width, uint height, InnerITKImage::PixelType* data);

    InnerITKImage::Pointer getPointer() const;
    ITKImage clone() const;

    static ITKImage read(std::string image_file_path);
    void write(std::string image_file_path);

    bool isNull() const;

    void foreachPixel(std::function<void(uint x, uint y, PixelType pixel)> callback) const;

    PixelType getPixel(uint x, uint y) const;
    void setPixel(uint x, uint y, PixelType value);

    void setEachPixel(std::function<PixelType(uint x, uint y)> pixel_fetcher);

    PixelType getPixel(InnerITKImage::IndexType index) const;

    uint getVisibleSliceIndex() const;
    void setVisibleSliceIndex(uint slice_index);
    uint getImageDimension() const;
    uint getDepth() const;

    PixelType minimum() const;
    PixelType maximum() const;
};

#endif // ITKIMAGE_H
