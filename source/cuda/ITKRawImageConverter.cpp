#include "ITKRawImageConverter.h"

ITKRawImageConverter::ITKRawImageConverter()
{
}


RawImage::Pointer ITKRawImageConverter::convert(
        ITKImage::Pointer itk_image)
{
    ITKImage::SizeType itk_size = itk_image->GetLargestPossibleRegion().GetSize();
    RawImage::Size cuda_size = convert(itk_size);
    RawImage::Pointer cuda_image = new RawImage(cuda_size);

    for(uint x=0; x < cuda_size.x; x++)
    {
        for(uint y=0; y< cuda_size.y; y++)
        {
                ITKImage::IndexType itk_index;
                itk_index[0] = x;
                itk_index[1] = y;
                itk_index[2] = 0;

                cuda_image->setPixel(x,y,
                                     itk_image->GetPixel(itk_index) );
        }
    }
    return cuda_image;
}

ITKRawImageConverter::ITKImage::Pointer ITKRawImageConverter::convert(
        RawImage::Pointer cuda_image)
{
    ITKImage::Pointer itk_image = ITKImage::New();
    itk_image->SetRegions(convert(cuda_image->size ));
    itk_image->Allocate();

    for(uint x=0; x< cuda_image->size.x; x++)
    {
        for(uint y=0; y< cuda_image->size.y; y++)
        {
            ITKImage::IndexType itk_index;
            itk_index[0] = x;
            itk_index[1] = y;
            itk_index[2] = 0;

            itk_image->SetPixel(itk_index,
                   cuda_image->getPixel(x,y));
        }
    }

    return itk_image;
}

ITKRawImageConverter::ITKImage::SizeType
ITKRawImageConverter::convert(RawImage::Size size)
{
    ITKImage::SizeType itk_size;
    itk_size[0] = size.x;
    itk_size[1] = size.y;
    itk_size[2] = 1; // TODO: RawImage3D
    return itk_size;
}

RawImage::Size ITKRawImageConverter::convert(ITKImage::SizeType size)
{
    RawImage::Size cuda_size = { size[0], size[1] };
    return cuda_size;
}
