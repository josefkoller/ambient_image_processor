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

#include "OriginSpacingWidget.h"
#include "ui_OriginSpacingWidget.h"

OriginSpacingWidget::OriginSpacingWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::OriginSpacingWidget)
{
    ui->setupUi(this);
}

OriginSpacingWidget::~OriginSpacingWidget()
{
    delete ui;
}

void OriginSpacingWidget::registerModule(ImageWidget* image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(image_widget, &ImageWidget::imageChanged,
            [this](ITKImage image) {
        if(image.isNull())
            return;

        this->ui->originXSpinbox->setValue(image.getPointer()->GetOrigin()[0]);
        this->ui->originYSpinbox->setValue(image.getPointer()->GetOrigin()[1]);
        this->ui->spacingXSpinbox->setValue(image.getPointer()->GetSpacing()[0]);
        this->ui->spacingYSpinbox->setValue(image.getPointer()->GetSpacing()[1]);

        if(ITKImage::ImageDimension > 2) {
            this->ui->originZSpinbox->setValue(image.getPointer()->GetOrigin()[2]);
            this->ui->spacingZSpinbox->setValue(image.getPointer()->GetSpacing()[2]);
        }
    });
}

void OriginSpacingWidget::on_performButton_clicked()
{
    this->processInWorkerThread();
}

ITKImage OriginSpacingWidget::processImage(ITKImage image)
{
    ITKImage::InnerITKImage::SpacingType spacing;
    spacing[0] = this->ui->spacingXSpinbox->value();
    spacing[1] = this->ui->spacingYSpinbox->value();
    ITKImage::InnerITKImage::PointType origin;
    origin[0] = this->ui->originXSpinbox->value();
    origin[1] = this->ui->originYSpinbox->value();

    if(ITKImage::ImageDimension > 2) {
        spacing[2] = this->ui->spacingZSpinbox->value();
        origin[2] = this->ui->originZSpinbox->value();
    }

    auto result = image.clone();
    result.getPointer()->SetOrigin(origin);
    result.getPointer()->SetSpacing(spacing);
    return result;
}
