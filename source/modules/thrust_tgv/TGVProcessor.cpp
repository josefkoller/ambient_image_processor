#include "TGVProcessor.h"


#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIteratorWithIndex.h>


DeviceImage* filterGPU(DeviceImage* f, const Pixel lambda, const uint iteration_count,
                       TGVProcessor::DeviceIterationFinished iteration_finished_callback);
HostImage* filterCPU(HostImage* f, const Pixel lambda, const uint iteration_count,
                     TGVProcessor::HostIterationFinished iteration_finished_callback);

TGVProcessor::TGVProcessor()
{
}

template<typename ThrustImage>
ThrustImage* TGVProcessor::convert(ITKImage itk_image)
{
    ThrustImage* image = new ThrustImage(itk_image.width, itk_image.height);

    itk_image.foreachPixel([image](uint x, uint y, ITKImage::PixelType pixel) {
        image->setPixel(x,y,pixel);
    });
    return image;
}

template<typename ThrustImage>
ITKImage TGVProcessor::convert(ThrustImage* image)
{
    ITKImage itk_image = ITKImage(image->width, image->height);

    itk_image.setEachPixel([&image](uint x, uint y) {
        return image->getPixel(x,y);
    });

    return itk_image;
}

ITKImage TGVProcessor::processTVL2GPU(ITKImage input_image,
   const Pixel lambda, const uint iteration_count, IterationFinished iteration_finished_callback)
{
    DeviceImage* f = convert<DeviceImage>(input_image);
    DeviceImage* u = filterGPU(f, lambda, iteration_count,
        [iteration_finished_callback](uint index, uint count, DeviceImage* u) {
            ITKImage itk_u = convert(u);
            iteration_finished_callback(index, count, itk_u);
    });

    delete f;

    ITKImage result = convert(u);
    delete u;
    return result;
}

ITKImage TGVProcessor::processTVL2CPU(ITKImage input_image,
   const Pixel lambda, const uint iteration_count, IterationFinished iteration_finished_callback)
{
    HostImage* f = convert<HostImage>(input_image);
    HostImage* u = filterCPU(f, lambda, iteration_count,
         [iteration_finished_callback](uint index, uint count, HostImage* u) {
             ITKImage itk_u = convert(u);
             iteration_finished_callback(index, count, itk_u);
    });

    delete f;

    ITKImage result = convert(u);
    delete u;
    return result;
}
