#include "TGVProcessor.h"


#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIteratorWithIndex.h>

#include "thrust/system_error.h"

DeviceThrustImage* filterGPU(DeviceThrustImage* f, const Pixel lambda, const uint iteration_count,
                       TGVProcessor::DeviceIterationFinished iteration_finished_callback);
HostThrustImage* filterCPU(HostThrustImage* f, const Pixel lambda, const uint iteration_count,
                     TGVProcessor::HostIterationFinished iteration_finished_callback);

TGVProcessor::TGVProcessor()
{
}

template<typename ThrustImage>
ThrustImage* TGVProcessor::convert(ITKImage itk_image)
{
    HostPixelVector host_vector = HostPixelVector(itk_image.voxel_count);

    itk_image.foreachPixel([&itk_image, &host_vector](uint x, uint y, uint z, ITKImage::PixelType pixel) {
        uint index = itk_image.linearIndex(x,y,z);
        host_vector[index] = pixel;
    });

    typename ThrustImage::Vector vector = host_vector; // may copy the whole vector to the graphic card storage

    return new ThrustImage(itk_image.width, itk_image.height, itk_image.depth, vector);
}

template<typename ThrustImage>
ITKImage TGVProcessor::convert(ThrustImage* image)
{
    ITKImage itk_image = ITKImage(image->width, image->height, image->depth);

    HostPixelVector host_vector = image->pixel_rows; // may copy the whole vector from the graphic card storage

    itk_image.setEachPixel([&itk_image, &host_vector](uint x, uint y, uint z) {
        uint index = itk_image.linearIndex(x,y,z);
        return host_vector[index];
    });

    return itk_image;
}

ITKImage TGVProcessor::processTVL2GPU(ITKImage input_image,
   const Pixel lambda, const uint iteration_count, IterationFinished iteration_finished_callback)
{
    try
    {
        DeviceThrustImage* f = convert<DeviceThrustImage>(input_image);

        DeviceThrustImage* u = filterGPU(f, lambda, iteration_count,
            [iteration_finished_callback](uint index, uint count, DeviceThrustImage* u) {
                ITKImage itk_u = convert(u);
                iteration_finished_callback(index, count, itk_u);
        });
        delete f;

        ITKImage result = convert(u);
        delete u;
        return result;

    }
    catch(thrust::system::system_error error)
    {
        std::cerr << "error: " << error.what() << std::endl;
        throw error;
    }
    return ITKImage();
}


ITKImage TGVProcessor::processTVL2CPU(ITKImage input_image,
   const Pixel lambda, const uint iteration_count, IterationFinished iteration_finished_callback)
{
    HostThrustImage* f = convert<HostThrustImage>(input_image);
    HostThrustImage* u = filterCPU(f, lambda, iteration_count,
                                   nullptr);
                                   /*
         [iteration_finished_callback](uint index, uint count, HostThrustImage* u) {
             ITKImage itk_u = convert(u);
             iteration_finished_callback(index, count, itk_u);
    });
                                   */

    delete f;

    ITKImage result = convert(u);
    delete u;
    return result;
}
