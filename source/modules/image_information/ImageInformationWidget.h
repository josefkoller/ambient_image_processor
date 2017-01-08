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

#ifndef IMAGEINFORMATIONWIDGET_H
#define IMAGEINFORMATIONWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"

namespace Ui {
class ImageInformationWidget;
}

class ImageInformationWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit ImageInformationWidget(QString title, QWidget *parent = 0);
    ~ImageInformationWidget();

private:
    Ui::ImageInformationWidget *ui;

public:
    virtual void registerModule(ImageWidget* image_widget);

    void getCVAndTV(ITKImage::PixelType& cv, ITKImage::PixelType& tv);
private slots:
    void collectInformation(ITKImage image);
    void on_copy_to_clipboard_button_clicked();
};

#endif // IMAGEINFORMATIONWIDGET_H
