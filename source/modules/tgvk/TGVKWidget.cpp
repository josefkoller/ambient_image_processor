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

#include "TGVKWidget.h"
#include "ui_TGVKWidget.h"

#include <QGroupBox>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QRadioButton>


TGVKWidget::TGVKWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::TGVKWidget)
{
    ui->setupUi(this);
    this->updateAlpha();
}

TGVKWidget::~TGVKWidget()
{
    delete ui;
}

void TGVKWidget::updateAlpha()
{
    delete this->ui->alpha_groupbox;
    this->ui->alpha_groupbox = new QGroupBox("Alpha", this->ui->parameterGroupBox);
    this->ui->alpha_groupbox->setLayout(new QVBoxLayout());
    this->ui->parameterGroupBox->layout()->addWidget(this->ui->alpha_groupbox);

    this->alpha_spinboxes.clear();

    const int order = this->ui->order_spinbox->value();
    for(int k = 0; k < order; k++)
    {
        this->addAlpha(k);
    }
}

void TGVKWidget::addAlpha(uint index)
{
    auto alpha_groupbox = new QGroupBox(this->ui->alpha_groupbox);
    this->ui->alpha_groupbox->layout()->addWidget(alpha_groupbox);
    alpha_groupbox->setLayout(new QHBoxLayout());

    auto spinbox = new QDoubleSpinBox(alpha_groupbox);
    this->alpha_spinboxes.push_back(spinbox);

    spinbox->setMaximum(1e5);
    spinbox->setValue(index + 1);
    spinbox->setSingleStep(0.01);
    alpha_groupbox->layout()->addWidget(spinbox);
    alpha_groupbox->setTitle("Alpha" + QString::number(index));
}

ITKImage TGVKWidget::processImage(ITKImage image)
{
    const uint order = this->ui->order_spinbox->value();
    ITKImage::PixelType* alpha = new ITKImage::PixelType[order];
    for(int k = 0; k < order; k++)
        alpha[k] = this->alpha_spinboxes.at(k)->value();

    const float lambda = this->ui->lambda_spinbox->value();
    const uint iteration_count = this->ui->iteration_count_spinbox->value();
    const uint paint_iteration_interval = this->ui->paint_iteration_interval_spinbox->value();

    auto result = TGVKProcessor::processTGVKL1GPUCuda(
                image,
                lambda, order, const_cast<ITKImage::PixelType*>(alpha),
                iteration_count,
                paint_iteration_interval, this->iteration_finished_callback);
    delete[] alpha;

    return result;
}


void TGVKWidget::on_perform_button_clicked()
{
    this->stop_after_next_iteration = false;
    this->processInWorkerThread();
    this->ui->stop_button->setEnabled(true);
}

void TGVKWidget::on_stop_button_clicked()
{
    this->stop_after_next_iteration = true;
}

void TGVKWidget::setIterationFinishedCallback(TGVKProcessor::IterationFinished iteration_finished_callback)
{
    this->iteration_finished_callback = [this, iteration_finished_callback]
            (uint iteration_index, uint iteration_count, ITKImage u){
        iteration_finished_callback(iteration_index, iteration_count, u);
        return this->stop_after_next_iteration;
    };
}

void TGVKWidget::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(this, &BaseModuleWidget::fireWorkerFinished,
            this, [this]() {
        this->ui->stop_button->setEnabled(false);
    });
}

void TGVKWidget::on_order_spinbox_editingFinished()
{
    this->updateAlpha();
}
