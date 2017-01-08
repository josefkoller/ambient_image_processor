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

#ifndef IMAGEVIEWCONTROLWIDGET_H
#define IMAGEVIEWCONTROLWIDGET_H

#include "BaseModuleWidget.h"

namespace Ui {
class ImageViewControlWidget;
}

class ImageViewControlWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    ImageViewControlWidget(QString title, QWidget *parent);
    ~ImageViewControlWidget();

private slots:
    void on_do_rescale_checkbox_toggled(bool checked);
    void on_do_multiply_checkbox_toggled(bool checked);
    void on_useWindowCheckbox_toggled(bool checked);
    void on_use_mask_module_checkbox_clicked(bool checked);

private:
    Ui::ImageViewControlWidget *ui;

signals:
    void doRescaleChanged(bool do_rescale);
    void doMultiplyChanged(bool do_multiply);
    void useWindowChanged(bool use_window);
    void useMaskModule(bool use_mask_module);
};

#endif // IMAGEVIEWCONTROLWIDGET_H
