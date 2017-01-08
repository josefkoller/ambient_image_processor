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

#include "BilateralFilterWidget.h"
#include "ui_BilateralFilterWidget.h"

#include "BilateralFilterProcessor.h"

BilateralFilterWidget::BilateralFilterWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::BilateralFilterWidget)
{
    ui->setupUi(this);
}

BilateralFilterWidget::~BilateralFilterWidget()
{
    delete ui;
}

void BilateralFilterWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}

ITKImage BilateralFilterWidget::processImage(ITKImage image)
{
    if(image.isNull())
        return ITKImage();

    float sigma_spatial_distance = this->ui->sigmaSpatialDistanceSpinbox->value();
    float sigma_intensity_distance = this->ui->sigmaIntensityDistanceSpinbox->value();
    int kernel_size = this->ui->kernelSizeSpinbox->value();

    return BilateralFilterProcessor::process( image,
                                                sigma_spatial_distance,
                                                sigma_intensity_distance,
                                                kernel_size);
}
