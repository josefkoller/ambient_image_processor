#ifndef CUDAIMAGEPROCESSOR_H
#define CUDAIMAGEPROCESSOR_H

#include "ITKImageProcessor.h"
#include "functional"

// see
// http://de.cppreference.com/w/cpp/utility/functional/function
// http://www.drdobbs.com/parallel/unified-memory-in-cuda-6-a-brief-overvie/240169095?pgno=2

#include "RawImage.h"

class CudaImageProcessor
{
private:
    CudaImageProcessor();
public:
    typedef ITKImageProcessor::ImageType  ITKImage;

    static RawImage* clone(RawImage* image);
    static void removeSensorSensitivity(ITKImage::Pointer f,
                                        double lambda,
                                        std::function<void(ITKImage::Pointer)> iteration_callback,
                                        std::function<void(ITKImage::Pointer)> finished_callback);
    static void removeSensorSensitivity(RawImage* f,
                                        double lambda,
                                        std::function<void(ITKImage::Pointer)> iteration_callback,
                                        std::function<void(ITKImage::Pointer)> finished_callback);
};

#endif // CUDAIMAGEPROCESSOR_H
