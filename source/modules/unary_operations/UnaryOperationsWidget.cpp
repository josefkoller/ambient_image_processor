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

#include "UnaryOperationsWidget.h"
#include "ui_UnaryOperationsWidget.h"

#include "CudaImageOperationsProcessor.h"
#include "RescaleIntensityProcessor.h"

UnaryOperationsWidget::UnaryOperationsWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::UnaryOperationsWidget)
{
    ui->setupUi(this);
}

UnaryOperationsWidget::~UnaryOperationsWidget()
{
    delete ui;
}

ITKImage UnaryOperationsWidget::processImage(ITKImage image)
{
    if(this->ui->invert_checkbox->isChecked())
        return CudaImageOperationsProcessor::invert(image);

    if(this->ui->binarize_checkbox->isChecked())
        return CudaImageOperationsProcessor::binarize(image);

    if(this->ui->dct_checkbox->isChecked())
        return CudaImageOperationsProcessor::cosineTransform(image);
    if(this->ui->idct_checkbox->isChecked())
        return CudaImageOperationsProcessor::inverseCosineTransform(image);

    if(this->ui->exp_checkbox->isChecked())
    {
        return CudaImageOperationsProcessor::exp(image);

        /*
        auto one = image.clone();
        one.setEachPixel([] (uint,uint,uint) { return 1.0; });
        return CudaImageOperationsProcessor::subtract(image, one); addConstant(image, -1);
        */
    }

    if(this->ui->log_checkbox->isChecked())
    {
        // image = RescaleIntensityProcessor::process(image, 1, 2);
        return CudaImageOperationsProcessor::log(image);
    }

    if(this->ui->div_grad_checkbox->isChecked())
    {
        return CudaImageOperationsProcessor::divGrad(image);
    }

    if(this->ui->remove_zero_frequency_checkbox->isChecked())
        return CudaImageOperationsProcessor::remove_zero_frequency(image);

    if(this->ui->rotate180_in_plane_checkbox->isChecked())
        return CudaImageOperationsProcessor::rotate180InPlane(image);
}

void UnaryOperationsWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}
