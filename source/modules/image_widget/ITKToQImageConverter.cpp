 #include "ITKToQImageConverter.h"


ITKToQImageConverter::ImageType::PixelType* ITKToQImageConverter::window_from = nullptr;
ITKToQImageConverter::ImageType::PixelType* ITKToQImageConverter::window_to = nullptr;

QImage* ITKToQImageConverter::convert(ImageType::Pointer itk_image, uint slice_index)
{
    ImageType::RegionType region = itk_image->GetLargestPossibleRegion();
    ImageType::SizeType size = region.GetSize();

    typedef itk::RescaleIntensityImageFilter<ImageType> RescaleFilter;
    RescaleFilter::Pointer rescale_filter = RescaleFilter::New();
    rescale_filter->SetOutputMinimum(0);
    rescale_filter->SetOutputMaximum(255);
    rescale_filter->SetInput( itk_image );
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

            if(InputDimension > 2)
                index[2] = slice_index;

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

QImage ITKToQImageConverter::convert_mask(MaskImage::Pointer itk_image)
{
    MaskImage::RegionType region = itk_image->GetLargestPossibleRegion();
    MaskImage::SizeType size = region.GetSize();

    typedef itk::RescaleIntensityImageFilter<MaskImage> RescaleFilter;
    RescaleFilter::Pointer rescale_filter = RescaleFilter::New();
    rescale_filter->SetOutputMinimum(0);
    rescale_filter->SetOutputMaximum(255);

    rescale_filter->SetInput( itk_image );

    rescale_filter->Update();
    MaskImage::Pointer rescaled_image = rescale_filter->GetOutput();

    QImage q_image( QSize(size[0], size[1]), QImage::Format_ARGB32);

    for(int x = 0; x < q_image.size().width(); x++)
    {
        for(int y = 0; y < q_image.size().height(); y++)
        {
            ImageType::IndexType index;
            index[0] = x;
            index[1] = y;
            index[2] = 0;
            int value = rescaled_image->GetPixel(index);

            QColor color(value, value, value);
            q_image.setPixel(x, y, color.rgb());
        }
    }
    return q_image;
}

void ITKToQImageConverter::setWindowFrom(ImageType::PixelType value)
{
    if(window_from == nullptr)
        window_from = new ImageType::PixelType;
    *window_from = value;
}

void ITKToQImageConverter::setWindowTo(ImageType::PixelType value)
{
    if(window_to == nullptr)
        window_to = new ImageType::PixelType;
    *window_to = value;
}
