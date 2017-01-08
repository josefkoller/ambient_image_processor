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

#ifndef NORMALIZESTATISTICWIDGET_H
#define NORMALIZESTATISTICWIDGET_H

#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"

namespace Ui {
class NormalizeStatisticWidget;
}

class NormalizeStatisticWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    NormalizeStatisticWidget(QString title, QWidget *parent);
    ~NormalizeStatisticWidget();

protected:
    ITKImage processImage(ITKImage image);

private slots:
    void on_load_button_clicked();

    void on_performButton_clicked();

private:
    Ui::NormalizeStatisticWidget *ui;

    ImageViewWidget* reference_image_widget;
public:
    void registerModule(ImageWidget* image_widget);
};

#endif // NORMALIZESTATISTICWIDGET_H
