#include "CrosshairModule.h"

CrosshairModule::CrosshairModule(QString title) :
    BaseModule(title),
    image(ITKImage::Null)
{
}


void CrosshairModule::registerModule(ImageWidget* image_widget)
{
    BaseModule::registerModule(image_widget);

    connect(image_widget, &ImageWidget::imageChanged,
            this, [this] (ITKImage image) {
        this->image = image;
    });

    connect(image_widget, &ImageWidget::mouseMoveOnImage,
            this, &CrosshairModule::mouseMoveOnImage);
}

void CrosshairModule::mouseMoveOnImage(Qt::MouseButtons button, QPoint position)
{
    if(position.x() < 0 || position.x() > this->image.width ||
            position.y() < 0 || position.y() > this->image.height )
        return;

    // showing pixel value...
    ITKImage::InnerITKImage::PixelType pixel_value = this->image.getPixel(position.x(), position.y());
    QString text = QString("pixel value at ") +
            QString::number(position.x()) +
            " | " +
            QString::number(position.y()) +
            " = " +
            QString::number(pixel_value);
    this->setStatusText(text);
}
