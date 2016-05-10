
#include "RawImage.h"
#include <stdio.h>
#include <cstdlib>

#include <cuda_runtime.h>

RawImage::RawImage(const Size size)
        :size(size),
          pixel_count(size.x * size.y)
{
    cudaMallocManaged(&this->pixel_pointer, sizeof(Pixel) * pixel_count);
}

RawImage::RawImage(const RawImage& cloning_image) :
    pixel_count(cloning_image.pixel_count),
    pixel_pointer(cloning_image.pixel_pointer),
    size(cloning_image.size)
{
    printf("RawImage copy constructor not implemented");
    exit(1);
}

RawImage::~RawImage()
{
    cudaFree(this->pixel_pointer);
}

void RawImage::setPixel(uint x, uint y, Pixel pixel_value)
{
    uint index = this->oneDimensionalIndex(x, y);
    this->setPixel(index, pixel_value);
}

void RawImage::setPixel(uint i, Pixel pixel_value)
{
    this->pixel_pointer[i] = pixel_value;
}

RawImage::Pixel RawImage::getPixel(uint i)
{
    return this->pixel_pointer[i];
}

RawImage::Pixel RawImage::getPixel(uint x, uint y)
{
    uint index = this->oneDimensionalIndex(x, y);
    return this->getPixel(index);
}

RawImage::uint RawImage::oneDimensionalIndex(uint x, uint y)
{
    return x + y * this->size.x;
}
