#include "TGVProcessor.h"


#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIteratorWithIndex.h>

#include "thrust/system_error.h"

DeviceThrustImage* filterGPU(DeviceThrustImage* f, const Pixel lambda, const uint iteration_count,
                             const uint paint_iteration_interval,
                             TGVProcessor::DeviceIterationFinished iteration_finished_callback);
HostThrustImage* filterCPU(HostThrustImage* f, const Pixel lambda, const uint iteration_count,
                           const uint paint_iteration_interval,
                           TGVProcessor::HostIterationFinished iteration_finished_callback);

#include <functional>

template<typename Pixel>
using IterationCallback = std::function<void(uint iteration_index, uint iteration_count, Pixel* u)>;

template<typename Pixel>
Pixel* tgv_launch(Pixel* f_host,
                  uint width, uint height, uint depth,
                  Pixel lambda,
                  uint iteration_count,
                  uint paint_iteration_interval,
                  IterationCallback<Pixel> iteration_finished_callback,
                  Pixel alpha0,
                  Pixel alpha1);

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

ITKImage TGVProcessor::processTVL2GPUThrust(ITKImage input_image,
                                      const Pixel lambda, const uint iteration_count,
                                      const uint paint_iteration_interval,
                                      IterationFinished iteration_finished_callback)
{
    try
    {
        DeviceThrustImage* f = convert<DeviceThrustImage>(input_image);

        DeviceThrustImage* u = filterGPU(f, lambda, iteration_count, paint_iteration_interval,
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

ITKImage TGVProcessor::processTVL2GPUCuda(ITKImage input_image,
                                          const Pixel lambda, const uint iteration_count,
                                          const uint paint_iteration_interval,
                                          IterationFinished iteration_finished_callback)
{
    Pixel* f = input_image.cloneToPixelArray();

    IterationCallback<Pixel> iteration_callback = [&input_image, iteration_finished_callback] (
            uint iteration_index, uint iteration_count, Pixel* u) {
        auto itk_u = ITKImage(input_image.width, input_image.height, input_image.depth, u);
        iteration_finished_callback(iteration_index, iteration_count, itk_u);
    };

    Pixel* u = tgv_launch<Pixel>(f,
                          input_image.width, input_image.height, input_image.depth,
                          lambda,
                          iteration_count,
                          paint_iteration_interval,
                          iteration_callback,
                          2, 1); // TODO alpha0, alpha1


    delete f;

    auto result = ITKImage(input_image.width, input_image.height, input_image.depth, u);
    delete u;
    return result;
}

ITKImage TGVProcessor::processTVL2CPU(ITKImage input_image,
                                      const Pixel lambda, const uint iteration_count,
                                      const uint paint_iteration_interval,
                                      IterationFinished iteration_finished_callback)
{
    HostThrustImage* f = convert<HostThrustImage>(input_image);
    HostThrustImage* u = filterCPU(f, lambda, iteration_count, paint_iteration_interval,
                                   [iteration_finished_callback](uint index, uint count, HostThrustImage* u) {
            ITKImage itk_u = convert(u);
            iteration_finished_callback(index, count, itk_u);
});

    delete f;

    ITKImage result = convert(u);
    delete u;
    return result;
}
