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

#include "TGV3Widget.h"
#include "ui_TGV3Widget.h"

TGV3Widget::TGV3Widget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::TGV3Widget)
{
    ui->setupUi(this);
}

TGV3Widget::~TGV3Widget()
{
    delete ui;
}

void TGV3Widget::on_perform_button_clicked()
{
    this->stop_after_next_iteration = false;
    this->processInWorkerThread();
    this->ui->stop_button->setEnabled(true);
}

void TGV3Widget::on_stop_button_clicked()
{
    this->stop_after_next_iteration = true;
}

void TGV3Widget::setIterationFinishedCallback(TGV3Processor::IterationFinished iteration_finished_callback)
{
    this->iteration_finished_callback = [this, iteration_finished_callback]
            (uint iteration_index, uint iteration_count, ITKImage u){
        iteration_finished_callback(iteration_index, iteration_count, u);
        return this->stop_after_next_iteration;
    };
}

void TGV3Widget::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(this, &BaseModuleWidget::fireWorkerFinished,
            this, [this]() {
        this->ui->stop_button->setEnabled(false);
    });
}


ITKImage TGV3Widget::processImage(ITKImage image)
{
    const float alpha0 = this->ui->alpha0_spinbox->value();
    const float alpha1 = this->ui->alpha1_spinbox->value();
    const float alpha2 = this->ui->alpha2_spinbox->value();
    const float lambda = this->ui->lambda_spinbox->value();
    const uint iteration_count = this->ui->iteration_count_spinbox->value();
    const uint paint_iteration_interval = this->ui->paint_iteration_interval_spinbox->value();

    return TGV3Processor::processTGV3L1GPUCuda(
                image,
                lambda, alpha0, alpha1, alpha2,
                iteration_count,
                paint_iteration_interval, this->iteration_finished_callback);
}
