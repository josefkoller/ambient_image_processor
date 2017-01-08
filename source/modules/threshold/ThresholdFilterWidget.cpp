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

#include "ThresholdFilterWidget.h"
#include "ui_ThresholdFilterWidget.h"

#include "ThresholdFilterProcessor.h"

ThresholdFilterWidget::ThresholdFilterWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::ThresholdFilterWidget)
{
    ui->setupUi(this);
}

ThresholdFilterWidget::~ThresholdFilterWidget()
{
    delete ui;
}

ITKImage ThresholdFilterWidget::processImage(ITKImage image)
{
    if(image.isNull())
        return ITKImage();

    auto lower_threshold_value = this->ui->lowerThresholdSpinbox->value();
    auto upper_threshold_value = this->ui->upperThresholdSpinbox->value();
    auto outside_pixel_value = this->ui->outsideSpinbox->value();

    return ThresholdFilterProcessor::process( image,
                                              lower_threshold_value,
                                              upper_threshold_value,
                                              outside_pixel_value);
}

void ThresholdFilterWidget::on_thresholdButton_clicked()
{
    this->processInWorkerThread();
}
