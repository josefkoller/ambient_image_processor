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

#ifndef BINARYOPERATIONSWIDGET_H
#define BINARYOPERATIONSWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"

namespace Ui {
class BinaryOperationsWidget;
}

class BinaryOperationsWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    BinaryOperationsWidget(QString title, QWidget *parent);
    ~BinaryOperationsWidget();

    void registerModule(ImageWidget *image_widget);
private slots:
    void on_load_button_clicked();

    void on_perform_button_clicked();

    void on_image2_offset_spinbox_valueChanged(double arg1);

    void on_image2_factor_spinbox_valueChanged(double arg1);

    void on_clearButton_clicked();

private:
    Ui::BinaryOperationsWidget *ui;
    ImageViewWidget* second_image_widget;

protected:
    virtual ITKImage processImage(ITKImage image);
};

#endif // BINARYOPERATIONSWIDGET_H
