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

#include "ExtractWidget.h"
#include "ui_ExtractWidget.h"

#include "ExtractProcessor.h"

ExtractWidget::ExtractWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::ExtractWidget)
{
    ui->setupUi(this);

    connect(this->ui->from_x_spinbox, SIGNAL(valueChanged(int)),
            this, SLOT(updateExtractedSizeLabel(int)));
    connect(this->ui->to_x_spinbox, SIGNAL(valueChanged(int)),
            this, SLOT(updateExtractedSizeLabel(int)));

    connect(this->ui->from_y_spinbox, SIGNAL(valueChanged(int)),
            this, SLOT(updateExtractedSizeLabel(int)));
    connect(this->ui->to_y_spinbox, SIGNAL(valueChanged(int)),
            this, SLOT(updateExtractedSizeLabel(int)));

    connect(this->ui->from_z_spinbox, SIGNAL(valueChanged(int)),
            this, SLOT(updateExtractedSizeLabel(int)));
    connect(this->ui->to_z_spinbox, SIGNAL(valueChanged(int)),
            this, SLOT(updateExtractedSizeLabel(int)));
}

ExtractWidget::~ExtractWidget()
{
    delete ui;
}

ITKImage ExtractWidget::processImage(ITKImage image)
{
    if(image.isNull())
        return ITKImage();

    return ExtractProcessor::process(
                image,
                ui->from_x_spinbox->value(),
                ui->to_x_spinbox->value(),
                ui->from_y_spinbox->value(),
                ui->to_y_spinbox->value(),
                ui->from_z_spinbox->value(),
                ui->to_z_spinbox->value()
                );
}

void ExtractWidget::registerModule(ImageWidget* image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(image_widget, &ImageWidget::imageChanged,
            this, &ExtractWidget::imageChanged);
}

void ExtractWidget::imageChanged(ITKImage image)
{
    if(image.isNull())
        return;

    int max_z = 0;
    if(image.getImageDimension() >= 3)
    {
        max_z = image.depth - 1;
    }
    int max_x = image.width - 1;
    int max_y = image.height - 1;
    this->ui->from_x_spinbox->setMaximum(max_x);
    this->ui->from_y_spinbox->setMaximum(max_y);
    this->ui->from_z_spinbox->setMaximum(max_z);
    this->ui->from_x_spinbox->setValue(0);
    this->ui->from_y_spinbox->setValue(0);
    this->ui->from_z_spinbox->setValue(0);
    this->ui->to_x_spinbox->setMaximum(max_x);
    this->ui->to_y_spinbox->setMaximum(max_y);
    this->ui->to_z_spinbox->setMaximum(max_z);
    this->ui->to_x_spinbox->setValue(max_x);
    this->ui->to_y_spinbox->setValue(max_y);
    this->ui->to_z_spinbox->setValue(max_z);

    this->updateExtractedSizeLabel(123);
}

void ExtractWidget::updateExtractedSizeLabel(int)
    {
    int from_x = ui->from_x_spinbox->value();
    int to_x = ui->to_x_spinbox->value();
    int size_x = to_x - from_x + 1;

    int from_y = ui->from_y_spinbox->value();
    int to_y = ui->to_y_spinbox->value();
    int size_y = to_y - from_y + 1;

    int from_z = ui->from_z_spinbox->value();
    int to_z = ui->to_z_spinbox->value();
    int size_z = to_z - from_z + 1;

    QString size_text = QString("%1x%2x%3").arg(
                QString::number(size_x),
                QString::number(size_y),
                QString::number(size_z));

    this->ui->extracted_region_size_label->setText(size_text);
}

void ExtractWidget::on_extract_button_clicked()
{
    this->processInWorkerThread();

}
