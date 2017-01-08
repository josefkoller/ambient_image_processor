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

#ifndef BSPLINEINTERPOLATIONWIDGET_H
#define BSPLINEINTERPOLATIONWIDGET_H

#include <QWidget>
#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"
#include "MaskWidget.h"

namespace Ui {
class BSplineInterpolationWidget;
}

class BSplineInterpolationWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    BSplineInterpolationWidget(QString title, QWidget *parent);
    ~BSplineInterpolationWidget();

    void registerModule(ImageWidget *image_widget);
private:
    Ui::BSplineInterpolationWidget *ui;

    MaskWidget::MaskFetcher mask_fetcher;

protected:
    virtual ITKImage processImage(ITKImage image);
private slots:
    void on_performButton_clicked();
};

#endif // BSPLINEINTERPOLATIONWIDGET_H
