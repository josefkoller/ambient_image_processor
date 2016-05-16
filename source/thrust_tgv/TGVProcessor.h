#ifndef TGVPROCESSOR_H
#define TGVPROCESSOR_H

#include <itkImage.h>
#include "Image.cuh"

class TGVProcessor
{
private:
    TGVProcessor();
public:
    typedef itk::Image<float> itkImage;
private:
    static Image* convert(itkImage::Pointer itk_image);
    static itkImage::Pointer convert(Image* image);
public:

    static itkImage::Pointer processTVL2(itkImage::Pointer input_image,
      const float lambda, const uint iteration_count);
};

#endif // TGVPROCESSOR_H
