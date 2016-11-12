 #include "ITKToQImageConverter.h"

#include "ExtractProcessor.h"
#include "CudaImageOperationsProcessor.h"
#include "ThresholdFilterProcessor.h"
#include "RescaleIntensityProcessor.h"

#include <QColor>

ITKImage::PixelType* ITKToQImageConverter::window_from = nullptr;
ITKImage::PixelType* ITKToQImageConverter::window_to = nullptr;


const QColor ITKToQImageConverter::lower_window_color = QColor(121, 152, 118);
const QColor ITKToQImageConverter::upper_window_color = QColor(156, 59, 78);
const QColor ITKToQImageConverter::outside_mask_color = QColor(148, 101, 83);

QImage* ITKToQImageConverter::convert(ITKImage itk_image,
                                      ITKImage mask,
                                      uint slice_index,
                                      bool do_rescale, bool do_multiply, bool use_window)
{
    ITKImage slice_image = ExtractProcessor::extract_slice(itk_image, slice_index);

    if(!mask.isNull())
        mask = ExtractProcessor::extract_slice(mask, slice_index);

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
        image_to_show = RescaleIntensityProcessor::process(rescale_source, 0, 255, mask).getPointer();
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

            ITKImage::PixelType non_rescaled_pixel_value = slice_image.getPixel(x, y, 0);

            // below window
            if(use_window && window_from != nullptr && non_rescaled_pixel_value < (*window_from))
            {
                color = lower_window_color;
            }
            // above window
            if(use_window && window_to != nullptr && non_rescaled_pixel_value > (*window_to))
            {
                color = upper_window_color;
            }

            // mask
            if(!mask.isNull() && mask.getPixel(x,y, slice_index) == 0) {
                color = outside_mask_color;
            }

            q_image->setPixel(x, y, color.rgb());
        }
    }

    if(invalid_pixel)
    {
        std::cout << "there are pixels < 0 or > 255" << std::endl;
    }

    // std::cout << "converted image slice " << slice_index << std::endl;

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
