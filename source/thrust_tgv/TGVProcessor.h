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
    template<typename ThrustImage>
    static ThrustImage* convert(itkImage::Pointer itk_image);

    template<typename ThrustImage>
    static itkImage::Pointer convert(ThrustImage* image);
public:

    static itkImage::Pointer processTVL2GPU(itkImage::Pointer input_image,
      const float lambda, const uint iteration_count);
    static itkImage::Pointer processTVL2CPU(itkImage::Pointer input_image,
      const float lambda, const uint iteration_count);
};

#endif // TGVPROCESSOR_H
