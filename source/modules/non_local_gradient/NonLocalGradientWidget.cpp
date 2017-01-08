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

#include "NonLocalGradientWidget.h"
#include "ui_NonLocalGradientWidget.h"

NonLocalGradientWidget::NonLocalGradientWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::NonLocalGradientWidget)
{
    ui->setupUi(this);

    this->ui->kernel_view->setSigma(this->ui->sigma_spinbox->value());
    this->ui->kernel_view->setKernelSize(this->ui->kernel_size_spinbox->value());
}

NonLocalGradientWidget::~NonLocalGradientWidget()
{
    delete ui;
}

void NonLocalGradientWidget::on_kernel_size_spinbox_valueChanged(int arg1)
{
    auto kernel_size = this->ui->kernel_size_spinbox->value();
    this->ui->kernel_view->setKernelSize(kernel_size);
}

void NonLocalGradientWidget::on_sigma_spinbox_valueChanged(double arg1)
{
    auto sigma = this->ui->sigma_spinbox->value();
    this->ui->kernel_view->setSigma(sigma);
}

void NonLocalGradientWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}

ITKImage NonLocalGradientWidget::processImage(ITKImage image)
{
    uint kernel_size = this->ui->kernel_size_spinbox->value();
    float kernel_sigma = this->ui->sigma_spinbox->value();

    ITKImage result_image = NonLocalGradientProcessor::process(
                image, kernel_size, kernel_sigma);

    return result_image;
}

float NonLocalGradientWidget::getKernelSigma() const
{
    return this->ui->sigma_spinbox->value();
}
uint NonLocalGradientWidget::getKernelSize() const
{
    return this->ui->kernel_size_spinbox->value();
}
