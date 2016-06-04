#include "CrosshairModule.h"

#include "ImageViewWidget.h"

CrosshairModule::CrosshairModule(QString title) :
    BaseModule(title),
    image(ITKImage::Null)
{
}


void CrosshairModule::registerModule(ImageViewWidget* image_view_widget, ImageWidget* image_widget)
{
    BaseModule::registerModule(image_widget);

    connect(image_view_widget, &ImageViewWidget::imageChanged,
            this, [this] (ITKImage image) {
        this->image = image;
    });

    connect(image_view_widget, &ImageViewWidget::mouseMoveOnImage,
            this, &CrosshairModule::mouseMoveOnImage);
}

void CrosshairModule::mouseMoveOnImage(Qt::MouseButtons button, ITKImage::Index cursor_index)
{
    if(this->image.isNull() || ! this->image.contains(cursor_index))
        return;

    ITKImage::InnerITKImage::PixelType pixel_value = this->image.getPixel(cursor_index);
    QString text = QString("pixel value at ") +
            ITKImage::indexToText(cursor_index) +
            " = " +
            QString::number(pixel_value);
    this->setStatusText(text);
}
