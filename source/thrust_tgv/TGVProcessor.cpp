#include "TGVProcessor.h"


#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIteratorWithIndex.h>

Image* filter(Image* f, const float lambda, const uint iteration_count);

TGVProcessor::TGVProcessor()
{
}

Image* TGVProcessor::convert(itkImage::Pointer itk_image)
{
    itkImage::SizeType size = itk_image->GetLargestPossibleRegion().GetSize();

    Image* image = new Image(size[0], size[1]);

    itk::ImageRegionConstIteratorWithIndex<itkImage> iterator(itk_image, itk_image->GetLargestPossibleRegion());
    while(!iterator.IsAtEnd())
    {
        itkImage::IndexType index = iterator.GetIndex();
        image->setPixel(index[0], index[1], iterator.Get());
        ++iterator;
    }
    return image;
}

TGVProcessor::itkImage::Pointer TGVProcessor::convert(Image* image)
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

TGVProcessor::itkImage::Pointer TGVProcessor::processTVL2(itkImage::Pointer input_image,
   const float lambda, const uint iteration_count)
{
    Image* f = convert(input_image);
    Image* u = filter(f, lambda, iteration_count);

    delete f;

    TGVProcessor::itkImage::Pointer result = convert(u);
    delete u;
    return result;
}
