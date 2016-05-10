#include "CudaImageProcessor.h"

#include "ITKRawImageConverter.h"

extern "C" void clone_image_launch(RawImage::Pointer source,
                                   RawImage::Pointer destination);

CudaImageProcessor::CudaImageProcessor()
{
}

RawImage::Pointer CudaImageProcessor::clone(RawImage::Pointer image)
{
    RawImage* image_clone = new RawImage(image->size);

    std::cout << "pixel pointer source: " << image->pixel_pointer << std::endl;
    std::cout << "pixel pointer clone: " << image_clone->pixel_pointer << std::endl;

    clone_image_launch(image, image_clone);

    return image_clone;
}

void CudaImageProcessor::removeSensorSensitivity(ITKImage::Pointer f,
                                    double lambda,
                                    std::function<void(ITKImage::Pointer)> iteration_callback,
                                    std::function<void(ITKImage::Pointer)> finished_callback)
{
    RawImage::Pointer f_cuda_managed = ITKRawImageConverter::convert(f);
    removeSensorSensitivity(f_cuda_managed,
                                  lambda,
                                  iteration_callback,
                                  finished_callback);
}



void CudaImageProcessor::removeSensorSensitivity(RawImage::Pointer f_cuda_managed,
                                    double lambda,
                                    std::function<void(ITKImage::Pointer)> iteration_callback,
                                    std::function<void(ITKImage::Pointer)> finished_callback)
{
    RawImage::Pointer u_cuda_managed = clone(f_cuda_managed);


    // TODO call external

    ITKImage::Pointer u = ITKRawImageConverter::convert(u_cuda_managed );

    finished_callback(u);

}














