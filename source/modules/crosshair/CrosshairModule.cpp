#include "CrosshairModule.h"

CrosshairModule::CrosshairModule(QString title) :
    BaseModule(title)
{
}


void CrosshairModule::registerModule(ImageWidget* image_widget)
{
    BaseModule::registerModule(image_widget);

    connect(image_widget, &ImageWidget::imageChanged,
            this, [this] (ITKImage::InnerITKImage::Pointer image) {
        this->image = image;
    });

    connect(image_widget, &ImageWidget::mouseMoveOnImage,
            this, &CrosshairModule::mouseMoveOnImage);
}

void CrosshairModule::mouseMoveOnImage(Qt::MouseButtons button, QPoint position)
{
    ITKImage::InnerITKImage::SizeType size = this->image->GetLargestPossibleRegion().GetSize();
    if(position.x() < 0 || position.x() > size[0] ||
            position.y() < 0 || position.y() > size[1] )
        return;

    ITKImage::InnerITKImage::IndexType index;
    index[0] = position.x();
    index[1] = position.y();
 //TODO   if(ITKImage::InnerITKImage::ImageDimension > 2)
 //      index[2] = this->slice_index;

    // showing pixel value...
    ITKImage::InnerITKImage::PixelType pixel_value = this->image->GetPixel(index);
    QString text = QString("pixel value at ") +
            QString::number(position.x()) +
            " | " +
            QString::number(position.y()) +
            " = " +
            QString::number(pixel_value);
    this->setStatusText(text);
}
