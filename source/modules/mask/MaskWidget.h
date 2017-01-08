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

#ifndef MASKWIDGET_H
#define MASKWIDGET_H

#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"

#include <functional>

namespace Ui {
class MaskWidget;
}

class MaskWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    MaskWidget(QString, QWidget *parent);
    ~MaskWidget();

    void registerModule(ImageWidget *image_widget);

    typedef std::function<ITKImage()> MaskFetcher;
    static MaskFetcher createMaskFetcher(ImageWidget* image_widget);

    void setMaskImage(ITKImage mask);
private slots:
    void on_load_mask_button_clicked();
    void on_enabled_checkbox_clicked();

private:
    Ui::MaskWidget *ui;
    ImageViewWidget* mask_view;

    ITKImage image;

    ITKImage getMask();

signals:
    void maskChanged(ITKImage mask);
};

#endif // MASKWIDGET_H
