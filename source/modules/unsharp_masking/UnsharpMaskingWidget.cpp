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

#include "UnsharpMaskingWidget.h"
#include "ui_UnsharpMaskingWidget.h"

#include "UnsharpMaskingProcessor.h"

UnsharpMaskingWidget::UnsharpMaskingWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::UnsharpMaskingWidget)
{
    ui->setupUi(this);
}

UnsharpMaskingWidget::~UnsharpMaskingWidget()
{
    delete ui;
}

void UnsharpMaskingWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}

ITKImage UnsharpMaskingWidget::processImage(ITKImage image)
{
    auto kernel_size = this->ui->kernel_size->value();
    auto kernel_sigma = this->ui->kernel_sigma->value();
    auto factor = this->ui->factor->value();

    return UnsharpMaskingProcessor::process(image, kernel_size, kernel_sigma, factor);
}

void UnsharpMaskingWidget::on_factor_valueChanged(double arg1)
{
    this->processInWorkerThread();
}

void UnsharpMaskingWidget::on_kernel_sigma_valueChanged(double arg1)
{
    this->processInWorkerThread();
}

void UnsharpMaskingWidget::on_kernel_size_valueChanged(int arg1)
{
    this->processInWorkerThread();
}
