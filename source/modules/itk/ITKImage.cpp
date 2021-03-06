/*
    Ambient Image Processor - A tool to perform several imaging tasks
    
    Copyright (C) 2016 Josef Koller

    https://github.com/josefkoller/ambient_image_processor    
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "ITKImage.h"

#include <itkImageDuplicator.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkCastImageFilter.h>

#include "RescaleIntensityProcessor.h"
#include "ThresholdFilterProcessor.h"

#include <iostream>

#include "OpenCVFile.h"

#include <vector>

#include "cuda_host_helper.cuh"

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


ITKImage ITKImage::read(std::string image_file_path)
{
    // if color file... load the v channel in HSV space
    auto image_file_path_lower = QString::fromStdString(image_file_path).toLower();

    bool load_hsv =
            image_file_path_lower.endsWith("png") ||
            image_file_path_lower.endsWith("jpg") ||
            image_file_path_lower.endsWith("jpeg") ||
            image_file_path_lower.endsWith("bmp");

    if(load_hsv)
        return read_hsv(image_file_path);

    typedef itk::ImageFileReader<InnerITKImage> FileReader;
    FileReader::Pointer reader = FileReader::New();
    reader->SetFileName(image_file_path);
    try
    {
        reader->Update();
    }
    catch (itk::ExceptionObject &exception)
    {
        /*
        std::cerr << "Exception thrown while reading the image file: " <<
                     image_file_path << std::endl;
        std::cerr << exception << std::endl;
        */
        return ITKImage(nullptr);
    }
    return ITKImage(reader->GetOutput());
}

void ITKImage::write_png(std::string image_file_path) const
{
    // writing 32bit png
    unsigned short MAX_PIXEL_VALUE = 65535;

    auto rescaled_image = RescaleIntensityProcessor::process(*this, 0, MAX_PIXEL_VALUE);

    typedef itk::Image<unsigned short, InnerITKImage::ImageDimension> PNGImage;
    typedef itk::CastImageFilter<InnerITKImage, PNGImage> CastFilter;
    CastFilter::Pointer cast_filter = CastFilter::New();
    cast_filter->SetInput(rescaled_image.getPointer());
    cast_filter->Update();

    typedef itk::ImageFileWriter<PNGImage> WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(image_file_path);
    writer->SetInput(cast_filter->GetOutput());

    try
    {
        writer->Update();

    }
    catch(itk::ExceptionObject exception)
    {
        std::cerr << exception << std::endl;
    }
}

void ITKImage::write_dicom(std::string image_file_path) const
{
    unsigned short MAX_PIXEL_VALUE = 4096;

    auto rescaled_image = RescaleIntensityProcessor::process(*this, 0, MAX_PIXEL_VALUE);

    typedef itk::Image<unsigned short, InnerITKImage::ImageDimension> DICOMImage;
    typedef itk::CastImageFilter<InnerITKImage, DICOMImage> CastFilter;
    CastFilter::Pointer cast_filter = CastFilter::New();
    cast_filter->SetInput(rescaled_image.getPointer());
    cast_filter->Update();

    typedef itk::ImageFileWriter<DICOMImage> WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(image_file_path);
    writer->SetInput(cast_filter->GetOutput());

    try
    {
        writer->Update();

    }
    catch(itk::ExceptionObject exception)
    {
        std::cerr << exception << std::endl;
    }
}

void ITKImage::write(std::string image_file_path) const
{
    if(QString::fromStdString(image_file_path).endsWith("png"))
    {
        this->write_png(image_file_path);
    }
    else if(QString::fromStdString(image_file_path).endsWith("dcm"))
    {
        this->write_dicom(image_file_path);
    }
    else
    {
        typedef itk::ImageFileWriter<InnerITKImage> WriterType;
        WriterType::Pointer writer = WriterType::New();
        writer->SetFileName(image_file_path);
        writer->SetInput(this->inner_image);
        writer->UseCompressionOn();
        try
        {
            writer->Update();
        }
        catch(itk::ExceptionObject exception)
        {
            std::cerr << exception << std::endl;
        }
    }
}

ITKImage ITKImage::read_hsv(std::string image_file_path)
{
    return OpenCVFile::read(image_file_path);
}

void ITKImage::write_hsv(std::string image_file_path) const
{
    auto thresholded_image = ThresholdFilterProcessor::clamp(*this, 0, 255);
    OpenCVFile::write_into_hsv_channel(thresholded_image, image_file_path);
}

bool ITKImage::isNull() const
{
    return this->inner_image.IsNull() || this->inner_image->GetDataReleased();
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
    return this->getPixel(index.x, index.y, index.z);
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
    this->setPixel(index.x, index.y, index.z, value);
}

void ITKImage::setPixel(uint linear_index, PixelType value)
{
    this->setPixel(ITKImage::linearTo3DIndex(linear_index), value);
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

void ITKImage::minimumAndMaximum(PixelType& minimum, PixelType& maximum) const
{
    minimum = 1e7;
    maximum = -1e7;

    if(this->isNull())
        return;

    this->foreachPixel([&minimum, &maximum](uint, uint, uint, PixelType value) {
        if(value > maximum)
            maximum = value;
        if(value < minimum)
            minimum = value;
    });
}

void ITKImage::minimumAndMaximumInsideMask(PixelType& minimum, PixelType& maximum,
                                           const ITKImage& mask) const
{
    minimum = 1e7;
    maximum = -1e7;

    if(this->isNull())
        return;

    this->foreachPixel([&minimum, &maximum, mask](uint x, uint y, uint z, PixelType value) {
        if(mask.getPixel(x,y,z) == 0)
            return;

        if(value > maximum)
            maximum = value;
        if(value < minimum)
            minimum = value;
    });
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

ITKImage::Index ITKImage::linearTo3DIndex(uint linear_index) const
{
    const int z = std::floor(linear_index / (width*height));
    int index_rest = linear_index - z * (width*height);
    const int y = floorf(index_rest / width);
    index_rest = index_rest - y * width;
    const int x = index_rest;
    return {x, y, z};
}

bool ITKImage::contains(Index index) const
{
    return this->inner_image->GetLargestPossibleRegion().IsInside(index);
}

bool ITKImage::contains(PixelIndex index) const
{
    return index.z >= 0 && index.z < depth &&
           index.x >= 0 && index.x < width &&
           index.y >= 0 && index.y < height;
}


ITKImage::PixelType* ITKImage::cloneToPixelArray() const
{
    if(this->isNull())
        return nullptr;

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


ITKImage::PixelType* ITKImage::cloneToCudaPixelArray() const
{
    if(this->isNull())
        return nullptr;

    ITKImage::PixelType* clone = cudaMalloc<ITKImage::PixelType>(this->voxel_count);
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

void ITKImage::setOriginAndSpacingOf(const ITKImage& source_image)
{
    InnerITKImage::SpacingType spacing = source_image.getPointer()->GetSpacing();
    InnerITKImage::PointType origin = source_image.getPointer()->GetOrigin();

    this->getPointer()->SetSpacing(spacing);
    this->getPointer()->SetOrigin(origin);
}

bool ITKImage::hasSameSize(const ITKImage& other_image)
{
    return this->width == other_image.width &&
            this->height == other_image.height &&
            this->depth == other_image.depth;
}
