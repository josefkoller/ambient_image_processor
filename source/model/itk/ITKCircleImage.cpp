#include "ITKCircleImage.h"

#include <itkImageRegionIteratorWithIndex.h>

ITKImage::Pointer ITKCircleImage::create(CircleImage circle_image)
{
    ITKImage::Pointer itk_image = ITKImage::New();
    ITKImage::SizeType size;
    size[0] = circle_image.width;
    size[1] = circle_image.height;

    itk_image->SetRegions(size);
    itk_image->Allocate();

    itk::ImageRegionIteratorWithIndex<ITKImage> iterator(itk_image,
                     itk_image->GetLargestPossibleRegion());
    while(!iterator.IsAtEnd())
    {
        ITKImage::IndexType itk_index = iterator.GetIndex();
        uint index = itk_index[0] + itk_index[1] * circle_image.width;
        float pixel = circle_image.pixels[index];
        iterator.Set(pixel);
        ++iterator;
    }
    return itk_image;
}
