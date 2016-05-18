#include "ITKImage.h"

#include <itkImageDuplicator.h>
#include <itkImageFileReader.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIteratorWithIndex.h>

#include <iostream>

ITKImage::ITKImage(uint width, uint height) : width(width), height(height)
{
    this->inner_image = InnerITKImage::New();
    InnerITKImage::SizeType size;
    size[0] = this->width;
    size[1] = this->height;
    this->inner_image->SetRegions(size);
    this->inner_image->Allocate();
}

ITKImage::ITKImage(InnerITKImage::Pointer inner_image) : inner_image(inner_image),
    width(inner_image->GetLargestPossibleRegion().GetSize()[0]),
    height(inner_image->GetLargestPossibleRegion().GetSize()[1])
{
}

ITKImage::ITKImage(uint width, uint height, InnerITKImage::PixelType* data) : ITKImage(width, height)
{
    itk::ImageRegionIteratorWithIndex<InnerITKImage> iteration(this->inner_image,
        this->inner_image->GetLargestPossibleRegion());
    while(!iteration.IsAtEnd())
    {
        InnerITKImage::IndexType index = iteration.GetIndex();
        int i = index[0] + index[1] * this->width;

        iteration.Set(data[i]);

        ++iteration;
    }
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


ITKImage ITKImage::read(std::string image_file_path)
{
    typedef itk::ImageFileReader<InnerITKImage> FileReader;
    FileReader::Pointer reader = FileReader::New();
    reader->SetFileName(image_file_path);
    try
    {
        reader->Update();
    }
    catch (itk::ExceptionObject &exception)
    {
        std::cerr << "Exception thrown while reading the image file: " <<
                     image_file_path << std::endl;
        std::cerr << exception << std::endl;
        return ITKImage(nullptr);
    }

    // rescaling is necessary for png files...
    typedef itk::RescaleIntensityImageFilter<InnerITKImage,InnerITKImage> RescaleFilter;
    RescaleFilter::Pointer rescale_filter = RescaleFilter::New();
    rescale_filter->SetInput(reader->GetOutput());
    rescale_filter->SetOutputMinimum(0);
    rescale_filter->SetOutputMaximum(1);
    rescale_filter->Update();

    InnerITKImage::Pointer image = rescale_filter->GetOutput();
    image->DisconnectPipeline();
    return ITKImage(image);
}

void ITKImage::write(std::string image_file_path)
{
    // writing 32bit png
    unsigned short MAX_PIXEL_VALUE = 65535;
    typedef itk::RescaleIntensityImageFilter<InnerITKImage, InnerITKImage> RescaleFilter;
    RescaleFilter::Pointer rescale_filter = RescaleFilter::New();
    rescale_filter->SetInput(this->inner_image);
    rescale_filter->SetOutputMinimum(0);
    rescale_filter->SetOutputMaximum(MAX_PIXEL_VALUE);
    rescale_filter->Update();

    typedef itk::ImageFileWriter<InnerITKImage> WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(image_file_path);
    writer->SetInput(rescale_filter->GetOutput());
    writer->Update();
}

bool ITKImage::isNull() const
{
    return this->inner_image.IsNull();
}
