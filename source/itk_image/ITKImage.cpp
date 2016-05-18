#include "ITKImage.h"

#include <itkImageDuplicator.h>

ITKImage::ITKImage(uint width, uint height) : width(width), height(height)
{
    this->inner_image = InnerITKImage::New();
    InnerITKImage::SizeType size;
    size[0] = this->width;
    size[1] = this->height;
    this->inner_image->SetRegions(size);
    this->inner_image->Allocate();
}

ITKImage::InnerITKImage::Pointer ITKImage::getPointer() const
{
    return this->inner_image;
}

ITKImage::InnerITKImage::Pointer ITKImage::clone() const
{
    typedef itk::ImageDuplicator<InnerITKImage> Duplicator;
    typename Duplicator::Pointer duplicator = Duplicator::New();
    duplicator->SetInputImage(this->inner_image);
    duplicator->Update();

    InnerITKImage::Pointer clone = duplicator->GetOutput();
    clone->DisconnectPipeline();
    clone->SetSpacing(this->inner_image->GetSpacing());
    clone->SetOrigin(this->inner_image->GetOrigin());
    return clone;
}
