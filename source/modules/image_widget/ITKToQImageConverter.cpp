 #include "ITKToQImageConverter.h"

#include <itkRescaleIntensityImageFilter.h>
#include <QColor>

ITKImage::PixelType* ITKToQImageConverter::window_from = nullptr;
ITKImage::PixelType* ITKToQImageConverter::window_to = nullptr;

QImage* ITKToQImageConverter::convert(ITKImage itk_image)
{
    typedef ITKImage::InnerITKImage ImageType;

    ImageType::RegionType region = itk_image.getPointer()->GetLargestPossibleRegion();
    ImageType::SizeType size = region.GetSize();

    typedef itk::RescaleIntensityImageFilter<ITKImage::InnerITKImage> RescaleFilter;
    RescaleFilter::Pointer rescale_filter = RescaleFilter::New();
    rescale_filter->SetOutputMinimum(0);
    rescale_filter->SetOutputMaximum(255);
    rescale_filter->SetInput( itk_image.getPointer() );
    rescale_filter->Update();
    ImageType::Pointer rescaled_image = rescale_filter->GetOutput();

    QImage* q_image = new QImage( QSize(size[0], size[1]), QImage::Format_ARGB32);

    bool invalid_pixel = false;
    for(int x = 0; x < q_image->size().width(); x++)
    {
        for(int y = 0; y < q_image->size().height(); y++)
        {
            ImageType::IndexType index;
            index[0] = x;
            index[1] = y;

            if(itk_image.getImageDimension() > 2)
                index[2] = itk_image.getVisibleSliceIndex();

            int value = rescaled_image->GetPixel(index);

            if(value < 0)
            {
              //  std::cout << "pixel value [x=" << x << ", y=" << y << "] < 0, " << value << std::endl;
                value = 0;
                invalid_pixel = true;
            }
            if(value > 255)
            {
              //  std::cout << "pixel value [x=" << x << ", y=" << y << "] > 255, " << value << std::endl;
                value = 255;
                invalid_pixel = true;
            }

            QColor color(value, value, value);

            if(window_from != nullptr && value < (*window_from * 255))
            {
                color = QColor(0, 51, 253);
            }
            if(window_to != nullptr && value > (*window_to * 255))
            {
                color = QColor(206, 0, 0);
            }

            q_image->setPixel(x, y, color.rgb());
        }
    }

    if(invalid_pixel)
    {
        std::cout << "there are pixels < 0 or > 255" << std::endl;
    }

    std::cout << "converted image" << std::endl;

    return q_image;
}

void ITKToQImageConverter::setWindowFrom(ITKImage::PixelType value)
{
    if(window_from == nullptr)
        window_from = new ITKImage::PixelType;
    *window_from = value;
}

void ITKToQImageConverter::setWindowTo(ITKImage::PixelType value)
{
    if(window_to == nullptr)
        window_to = new ITKImage::PixelType;
    *window_to = value;
}
