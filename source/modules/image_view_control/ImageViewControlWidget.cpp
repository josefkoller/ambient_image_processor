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

#include "ImageViewControlWidget.h"
#include "ui_ImageViewControlWidget.h"

#include "ImageViewWidget.h"

ImageViewControlWidget::ImageViewControlWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::ImageViewControlWidget)
{
    ui->setupUi(this);
}

ImageViewControlWidget::~ImageViewControlWidget()
{
    delete ui;
}

void ImageViewControlWidget::on_do_rescale_checkbox_toggled(bool checked)
{
    emit this->doRescaleChanged(checked);
}

void ImageViewControlWidget::on_do_multiply_checkbox_toggled(bool checked)
{
    emit this->doMultiplyChanged(checked);
}

void ImageViewControlWidget::on_useWindowCheckbox_toggled(bool checked)
{
    emit this->useWindowChanged(checked);
}

void ImageViewControlWidget::on_use_mask_module_checkbox_clicked(bool checked)
{
    emit this->useMaskModule(checked);
}
