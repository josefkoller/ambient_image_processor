#include "TGVProcessor.h"


#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIteratorWithIndex.h>


DeviceImage* filterGPU(DeviceImage* f, const Pixel lambda, const uint iteration_count,
                       IterationFinished iteration_finished_callback);
HostImage* filterCPU(HostImage* f, const Pixel lambda, const uint iteration_count,
                     IterationFinished iteration_finished_callback);

TGVProcessor::TGVProcessor()
{
}

template<typename ThrustImage>
ThrustImage* TGVProcessor::convert(itkImage::Pointer itk_image)
{
    itkImage::SizeType size = itk_image->GetLargestPossibleRegion().GetSize();

    ThrustImage* image = new ThrustImage(size[0], size[1]);

    itk::ImageRegionConstIteratorWithIndex<itkImage> iterator(itk_image, itk_image->GetLargestPossibleRegion());
    while(!iterator.IsAtEnd())
    {
        itkImage::IndexType index = iterator.GetIndex();
        image->setPixel(index[0], index[1], iterator.Get());
        ++iterator;
    }
    return image;
}

template<typename ThrustImage>
TGVProcessor::itkImage::Pointer TGVProcessor::convert(ThrustImage* image)
{
    itkImage::Pointer itk_image = itkImage::New();

    itkImage::SizeType size;
    size[0] = image->width;
    size[1] = image->height;
    itk_image->SetRegions(size);
    itk_image->Allocate();

    itk::ImageRegionIteratorWithIndex<itkImage> iterator(itk_image, itk_image->GetLargestPossibleRegion());
    while(!iterator.IsAtEnd())
    {
        itkImage::IndexType index = iterator.GetIndex();
        iterator.Set(image->getPixel(index[0], index[1]));
        ++iterator;
    }
    return itk_image;
}

TGVProcessor::itkImage::Pointer TGVProcessor::processTVL2GPU(itkImage::Pointer input_image,
   const Pixel lambda, const uint iteration_count, IterationFinished iteration_finished_callback)
{
    DeviceImage* f = convert<DeviceImage>(input_image);
    DeviceImage* u = filterGPU(f, lambda, iteration_count, iteration_finished_callback);

    delete f;

    TGVProcessor::itkImage::Pointer result = convert(u);
    delete u;
    return result;
}

TGVProcessor::itkImage::Pointer TGVProcessor::processTVL2CPU(itkImage::Pointer input_image,
   const Pixel lambda, const uint iteration_count, IterationFinished iteration_finished_callback)
{
    HostImage* f = convert<HostImage>(input_image);
    HostImage* u = filterCPU(f, lambda, iteration_count, iteration_finished_callback);

    delete f;

    TGVProcessor::itkImage::Pointer result = convert(u);
    delete u;
    return result;
}
