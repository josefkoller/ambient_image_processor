 #include "ITKToQImageConverter.h"

#include "ExtractProcessor.h"
#include "CudaImageOperationsProcessor.h"
#include "ThresholdFilterProcessor.h"

#include <itkRescaleIntensityImageFilter.h>
#include <QColor>

ITKImage::PixelType* ITKToQImageConverter::window_from = nullptr;
ITKImage::PixelType* ITKToQImageConverter::window_to = nullptr;


const QColor ITKToQImageConverter::lower_window_color = QColor(0, 0, 0);
const QColor ITKToQImageConverter::upper_window_color = QColor(0, 0, 0);

QImage* ITKToQImageConverter::convert(ITKImage itk_image, uint slice_index,
                                      bool do_rescale, bool do_multiply, bool use_window)
{
    ITKImage slice_image = ExtractProcessor::extract_slice(itk_image, slice_index);

    ITKImage rescale_source;
    if(use_window && window_from != nullptr && window_to != nullptr) {
        ITKImage windowed_image = ThresholdFilterProcessor::process(slice_image,
          *window_from, 1e7, *window_from);
        windowed_image = ThresholdFilterProcessor::process(windowed_image,
          -1e7, *window_to, *window_to);
        rescale_source = windowed_image;
    } else {
        rescale_source = slice_image;
    }

    ITKImage::InnerITKImage::Pointer image_to_show;
    if(do_rescale)
    {
        typedef itk::RescaleIntensityImageFilter<ITKImage::InnerITKImage> RescaleFilter;
        RescaleFilter::Pointer rescale_filter = RescaleFilter::New();
        rescale_filter->SetOutputMinimum(0);
        rescale_filter->SetOutputMaximum(255);
        rescale_filter->SetInput( rescale_source.getPointer() );
        rescale_filter->Update();
        image_to_show = rescale_filter->GetOutput();
        image_to_show->DisconnectPipeline();
    }
    else
    {
        rescale_source = CudaImageOperationsProcessor::addConstant(rescale_source, slice_image.minimum());
        if(do_multiply)
            rescale_source = CudaImageOperationsProcessor::multiplyConstant(rescale_source, 255);
        image_to_show = rescale_source.getPointer();
    }

    QImage* q_image = new QImage( itk_image.width, itk_image.height, QImage::Format_ARGB32);

    bool invalid_pixel = false;
    for(int x = 0; x < q_image->size().width(); x++)
    {
        for(int y = 0; y < q_image->size().height(); y++)
        {
            ITKImage::InnerITKImage::IndexType index;
            index[0] = x;
            index[1] = y;

            if(itk_image.getImageDimension() > 2)
                index[2] = 0;

            int value = image_to_show->GetPixel(index);

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

            ITKImage::PixelType non_rescaled_pixel_value = rescale_source.getPixel(x, y, 0);

            if(use_window && window_from != nullptr && non_rescaled_pixel_value < (*window_from))
            {
                color = lower_window_color;
            }
            if(use_window && window_to != nullptr && non_rescaled_pixel_value > (*window_to))
            {
                color = upper_window_color;
            }

            q_image->setPixel(x, y, color.rgb());
        }
    }

    if(invalid_pixel)
    {
        std::cout << "there are pixels < 0 or > 255" << std::endl;
    }

    std::cout << "converted image slice " << slice_index << std::endl;

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
