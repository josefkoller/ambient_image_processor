#include "ITKImage.h"

#include <itkImageDuplicator.h>
#include <itkImageFileReader.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkCastImageFilter.h>
#include <iostream>

const ITKImage ITKImage::Null = ITKImage();

ITKImage::ITKImage(uint width, uint height, uint depth)
    : width(width), height(height), depth(depth),
      voxel_count(width*height*depth)
{
    this->inner_image = InnerITKImage::New();
    InnerITKImage::SizeType size;
    size[0] = this->width;
    size[1] = this->height;
    size[2] = this->depth;
    this->inner_image->SetRegions(size);
    this->inner_image->Allocate();
}

ITKImage::ITKImage(InnerITKImage::Pointer inner_image) : inner_image(inner_image),
    width(inner_image.IsNull() ? 0 : inner_image->GetLargestPossibleRegion().GetSize()[0]),
    height(inner_image.IsNull() ? 0 : inner_image->GetLargestPossibleRegion().GetSize()[1]),
    depth(inner_image.IsNull() ? 0 : inner_image->GetLargestPossibleRegion().GetSize()[2])
{
    if(inner_image.IsNotNull())
        inner_image->DisconnectPipeline();

    this->voxel_count = width*height*depth;
}

ITKImage::ITKImage() : ITKImage(nullptr)
{
}

ITKImage::ITKImage(const ITKImage& origin) :
    ITKImage(origin.inner_image)
{
}

ITKImage::ITKImage(uint width, uint height, uint depth, InnerITKImage::PixelType* data) :
    ITKImage(width, height, depth)
{
    itk::ImageRegionIteratorWithIndex<InnerITKImage> iteration(this->inner_image,
        this->inner_image->GetLargestPossibleRegion());
    while(!iteration.IsAtEnd())
    {
        InnerITKImage::IndexType index = iteration.GetIndex();
        int i = index[2] * this->width*this->height + (index[0] + index[1] * this->width);

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
    if(this->isNull())
        return ITKImage();

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


ITKImage ITKImage::read(std::string image_file_path, bool rescale)
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

    if(rescale)
    {
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
    InnerITKImage::Pointer image = reader->GetOutput();
    image->DisconnectPipeline();
    return ITKImage(image);
}

void ITKImage::write(std::string image_file_path)
{
    if(QString::fromStdString(image_file_path).endsWith("png"))
    {
        // writing 32bit png
        unsigned short MAX_PIXEL_VALUE = 65535;
        typedef itk::RescaleIntensityImageFilter<InnerITKImage, InnerITKImage> RescaleFilter;
        RescaleFilter::Pointer rescale_filter = RescaleFilter::New();
        rescale_filter->SetInput(this->inner_image);
        rescale_filter->SetOutputMinimum(0);
        rescale_filter->SetOutputMaximum(MAX_PIXEL_VALUE);
        rescale_filter->Update();

        typedef itk::Image<unsigned short, InnerITKImage::ImageDimension> PNGImage;
        typedef itk::CastImageFilter<InnerITKImage, PNGImage> CastFilter;
        CastFilter::Pointer cast_filter = CastFilter::New();
        cast_filter->SetInput(rescale_filter->GetOutput());
        cast_filter->Update();

        typedef itk::ImageFileWriter<PNGImage> WriterType;
        WriterType::Pointer writer = WriterType::New();
        writer->SetFileName(image_file_path);
        writer->SetInput(cast_filter->GetOutput());
        writer->Update();
    }
    else
    {
        typedef itk::ImageFileWriter<InnerITKImage> WriterType;
        WriterType::Pointer writer = WriterType::New();
        writer->SetFileName(image_file_path);
        writer->SetInput(this->inner_image);
        writer->Update();
    }
}

bool ITKImage::isNull() const
{
    return this->inner_image.IsNull();
}

void ITKImage::foreachPixel(std::function<void(uint x, uint y, uint z, PixelType pixel)> callback) const
{
    if(this->inner_image.IsNull())
        return;

    itk::ImageRegionConstIteratorWithIndex<InnerITKImage> iterator(
                this->inner_image, this->inner_image->GetLargestPossibleRegion());
    while(!iterator.IsAtEnd())
    {
        InnerITKImage::IndexType index = iterator.GetIndex();

        callback(index[0], index[1], index[2], iterator.Get());

        ++iterator;
    }
}

ITKImage::PixelType ITKImage::getPixel(uint x, uint y, uint z) const
{
    ITKImage::InnerITKImage::IndexType index;
    index[0] = x;
    index[1] = y;
    index[2] = z;

    return this->getPixel(index);
}

ITKImage::PixelType ITKImage::getPixel(InnerITKImage::IndexType index) const
{
    return this->inner_image->GetPixel(index);
}

ITKImage::PixelType ITKImage::getPixel(PixelIndex index) const
{
    return this->getPixel(index.toITKIndex());
}

void ITKImage::setPixel(uint x, uint y, uint z, PixelType value)
{
    ITKImage::InnerITKImage::IndexType index;
    index[0] = x;
    index[1] = y;
    index[2] = z;
    this->setPixel(index, value);
}

void ITKImage::setPixel(Index index, PixelType value)
{
    return this->inner_image->SetPixel(index, value);
}

void ITKImage::setPixel(PixelIndex index, PixelType value)
{
    this->setPixel(index.toITKIndex(), value);
}

void ITKImage::setEachPixel(std::function<PixelType(uint x, uint y, uint z)> pixel_fetcher)
{
    if(this->isNull())
        return;

    itk::ImageRegionIteratorWithIndex<InnerITKImage> iterator(
                this->inner_image, this->inner_image->GetLargestPossibleRegion());
    while(!iterator.IsAtEnd())
    {
        InnerITKImage::IndexType index = iterator.GetIndex();

        PixelType value = pixel_fetcher(index[0], index[1], index[2]);
        iterator.Set(value);

        ++iterator;
    }
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
    this->foreachPixel([&minimum](uint, uint, uint, PixelType value) {
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
    this->foreachPixel([&maximum](uint, uint, uint, PixelType value) {
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

uint ITKImage::linearIndex(Index index) const
{
    return this->linearIndex(index[0], index[1], index[2]);
}
uint ITKImage::linearIndex(uint x, uint y, uint z) const
{
    return z * this->width*this->height + (x + y * this->width);
}

bool ITKImage::contains(Index index) const
{
    return this->inner_image->GetLargestPossibleRegion().IsInside(index);
}

bool ITKImage::contains(PixelIndex index) const
{
    return this->inner_image->GetLargestPossibleRegion().IsInside(index.toITKIndex());
}


ITKImage::PixelType* ITKImage::cloneToPixelArray() const
{
    ITKImage::PixelType* clone = new ITKImage::PixelType[this->width*this->height*this->depth];
    if(clone == nullptr) {
        std::cerr << "memory allocation error in cloneToPixelArray: " <<
                     sizeof(ITKImage::PixelType)*this->width*this->height*this->depth <<
                     " bytes could not be reservated" << std::endl;
        return clone;
    }

    this->foreachPixel([clone, this](uint x, uint y, uint z, PixelType pixel) {
        uint i = this->linearIndex(x,y,z);
        clone[i] = pixel;
    });
    return clone;
}
ITKImage ITKImage::cloneSameSizeWithZeros() const
{
    ITKImage clone = ITKImage(this->width, this->height, this->depth);
    clone.setEachPixel([](uint, uint, uint) {
        return 0;
    });

    clone.getPointer()->SetOrigin(this->getPointer()->GetOrigin());
    clone.getPointer()->SetSpacing(this->getPointer()->GetSpacing());

    return clone;
}

uint ITKImage::linearIndex(Size size, ITKImage::InnerITKImage::IndexType index)
{
    return index[2] *size.x*size.y + index[0] + index[1] * size.x;
}
ITKImage::PixelType ITKImage::getPixel(PixelType* image_data, Size size, ITKImage::InnerITKImage::IndexType index)
{
    return image_data[linearIndex(size, index)];
}

void ITKImage::setPixel(PixelType* image_data, Size size, ITKImage::InnerITKImage::IndexType index, PixelType value)
{
    image_data[linearIndex(size, index)] = value;
}
