#ifndef ITKIMAGE_H
#define ITKIMAGE_H

#include <itkImage.h>
#include <string>

#include <QPoint>
#include <QString>

class ITKImage
{
public:
    typedef unsigned int uint;
    static const uint ImageDimension = 3;
    typedef double PixelType;

    typedef itk::Image<PixelType, ImageDimension> InnerITKImage;
    typedef InnerITKImage::IndexType Index;

    uint width;
    uint height;
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

    uint getImageDimension() const;
    uint getDepth() const;

    PixelType minimum() const;
    PixelType maximum() const;

    static ITKImage Null;

    static Index indexFromPoint(QPoint point, uint slice_index);
    static QPoint pointFromIndex(Index index);
    static QString indexToText(Index index);
};

#endif // ITKIMAGE_H
