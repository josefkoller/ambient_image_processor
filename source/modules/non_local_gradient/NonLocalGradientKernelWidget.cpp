#include "NonLocalGradientKernelWidget.h"
#include "ui_NonLocalGradientKernelWidget.h"

#include <QPainter>

#include <iostream>

NonLocalGradientKernelWidget::NonLocalGradientKernelWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::NonLocalGradientKernelWidget)
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
