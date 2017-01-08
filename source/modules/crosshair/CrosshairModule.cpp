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
