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

#include "NonLocalGradientKernelWidget.h"
#include "ui_NonLocalGradientKernelWidget.h"

#include <QPainter>

#include <iostream>

NonLocalGradientKernelWidget::NonLocalGradientKernelWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::NonLocalGradientKernelWidget),
    kernel_size(3),
    sigma(1)
{
    ui->setupUi(this);
    this->createKernelImage();
}

NonLocalGradientKernelWidget::~NonLocalGradientKernelWidget()
{
    delete ui;
}


void NonLocalGradientKernelWidget::setSigma(float sigma)
{
    if(sigma == this->sigma)
        return;

    this->sigma = sigma;
    this->createKernelImage();
}

void NonLocalGradientKernelWidget::setKernelSize(uint kernel_size)
{
    if(kernel_size == this->kernel_size)
        return;
    this->kernel_size = kernel_size;
    this->createKernelImage();
}

void NonLocalGradientKernelWidget::createKernelImage()
{
    this->kernel_image = QImage(this->kernel_size, this->kernel_size, QImage::Format_ARGB32);

    uint center = std::floor(this->kernel_size / 2.0f);
    for(uint x = 0; x < this->kernel_size; x++)
    {
        for(uint y = 0; y < this->kernel_size; y++)
        {
            uint xr = x - center;
            uint yr = y - center;
            float radius = std::sqrt(xr*xr + yr*yr);
            float value = std::exp(-radius*radius / this->sigma);

            value = value * 255;
//            std::cout << "value: " << value << std::endl;
            QColor color = QColor(value,value,value);
            this->kernel_image.setPixel(x,y,color.rgb());
        }
    }
    this->repaint();
}

void NonLocalGradientKernelWidget::paintEvent(QPaintEvent *event)
{
    QWidget::paintEvent(event);

    QPainter painter(this);
    painter.drawImage(0,0,this->kernel_image);
}
