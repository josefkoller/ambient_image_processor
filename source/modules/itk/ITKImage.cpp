#include "ITKImage.h"

#include <itkImageDuplicator.h>
#include <itkImageFileReader.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIteratorWithIndex.h>

#include <iostream>

ITKImage ITKImage::Null = ITKImage();

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
    width(inner_image.IsNull() ? 0 : inner_image->GetLargestPossibleRegion().GetSize()[0]),
    height(inner_image.IsNull() ? 0 : inner_image->GetLargestPossibleRegion().GetSize()[1])
{
    if(inner_image.IsNotNull())
        inner_image->DisconnectPipeline();
}

ITKImage::ITKImage() : ITKImage(nullptr)
{
}

ITKImage::ITKImage(const ITKImage& origin) :
    ITKImage(origin.inner_image)
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

ITKImage ITKImage::clone() const
{
    typedef itk::ImageDuplicator<InnerITKImage> Duplicator;
    typename Duplicator::Pointer duplicator = Duplicator::New();
    duplicator->SetInputImage(this->inner_image);
    duplicator->Update();

    InnerITKImage::Pointer clone = duplicator->GetOutput();
    clone->DisconnectPipeline();
    clone->SetSpacing(this->inner_image->GetSpacing());
    clone->SetOrigin(this->inner_image->GetOrigin());

    return ITKImage(clone);
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

void ITKImage::foreachPixel(std::function<void(uint x, uint y, PixelType pixel)> callback) const
{
    itk::ImageRegionConstIteratorWithIndex<InnerITKImage> iterator(
                this->inner_image, this->inner_image->GetLargestPossibleRegion());
    while(!iterator.IsAtEnd())
    {
        InnerITKImage::IndexType index = iterator.GetIndex();

        callback(index[0], index[1], iterator.Get());

        ++iterator;
    }
}

ITKImage::PixelType ITKImage::getPixel(uint x, uint y) const
{
    ITKImage::InnerITKImage::IndexType index;
    index[0] = x;
    index[1] = y;

    if(this->getImageDimension() > 2)
        index[2] = 0; // TODO make member getPixel(x,y,z)

    return this->getPixel(index);
}

ITKImage::PixelType ITKImage::getPixel(InnerITKImage::IndexType index) const
{
    return this->inner_image->GetPixel(index);
}

void ITKImage::setPixel(uint x, uint y, PixelType value)
{
    ITKImage::InnerITKImage::IndexType index;
    index[0] = x;
    index[1] = y;

    if(this->getImageDimension() > 2)
        index[2] = 0;  // TODO make member getPixel(x,y,z)

    return this->inner_image->SetPixel(index, value);
}

void ITKImage::setEachPixel(std::function<PixelType(uint x, uint y)> pixel_fetcher)
{
    itk::ImageRegionIteratorWithIndex<InnerITKImage> iterator(
                this->inner_image, this->inner_image->GetLargestPossibleRegion());
    while(!iterator.IsAtEnd())
    {
        InnerITKImage::IndexType index = iterator.GetIndex();

        PixelType value = pixel_fetcher(index[0], index[1]);
        iterator.Set(value);

        ++iterator;
    }
}

uint ITKImage::getDepth() const
{
    if(this->getImageDimension() <= 2)
        return 1;
    return this->inner_image->GetLargestPossibleRegion().GetSize()[2];
}

uint ITKImage::getImageDimension() const
{
    return this->inner_image->GetImageDimension();
}

ITKImage::PixelType ITKImage::minimum() const
{
    if(this->isNull())
        return 0;

    PixelType minimum = 1e7;
    this->foreachPixel([&minimum](uint x, uint y, PixelType value) {
        if(value < minimum)
            minimum = value;
    });
    return minimum;
}

ITKImage::PixelType ITKImage::maximum() const
{
    if(this->isNull())
        return 0;

    PixelType maximum = -1e7;
    this->foreachPixel([&maximum](uint x, uint y, PixelType value) {
        if(value > maximum)
            maximum = value;
    });
    return maximum;
}

ITKImage::Index ITKImage::indexFromPoint(QPoint point, uint slice_index)
{
    ITKImage::Index index;
    index[0] = point.x();
    index[1] = point.y();
    if(index.Dimension > 2)
        index[2] = slice_index;
    return index;
}

QPoint ITKImage::pointFromIndex(Index index)
{
    return QPoint(index[0], index[1]);
}

QString ITKImage::indexToText(Index index)
{
    return QString("%1 | %2 | %3").arg(
            QString::number(index[0]),
            QString::number(index[1]),
            QString::number(index[2]) );

}
