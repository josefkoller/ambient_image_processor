#ifndef TGVPROCESSOR_H
#define TGVPROCESSOR_H

#include <itkImage.h>
#include "Image.cuh"

typedef std::function<void(uint iteration_index, uint iteration_count)> IterationFinished;

class TGVProcessor
{
private:
    TGVProcessor();
public:
    typedef itk::Image<Pixel> itkImage;
private:
    template<typename ThrustImage>
    static ThrustImage* convert(itkImage::Pointer itk_image);

    template<typename ThrustImage>
    static itkImage::Pointer convert(ThrustImage* image);
public:

    static itkImage::Pointer processTVL2GPU(itkImage::Pointer input_image,
      const Pixel lambda, const uint iteration_count, IterationFinished iteration_finished_callback);
    static itkImage::Pointer processTVL2CPU(itkImage::Pointer input_image,
      const Pixel lambda, const uint iteration_count, IterationFinished iteration_finished_callback);
};

#endif // TGVPROCESSOR_H
