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

#include "MultiScaleRetinexScale.h"
#include "ui_MultiScaleRetinexScale.h"

#include <iostream>

MultiScaleRetinexScale::MultiScaleRetinexScale(QWidget *frame,
                                               MultiScaleRetinex::Scale* scale,
                                               const unsigned int index,
                                               std::function<void(unsigned int scale_index)> removeCallback) :
    QWidget(frame),
    ui(new Ui::MultiScaleRetinexScale),
    scale(scale),
    index(index)
{
    ui->setupUi(this);

    this->setIndex(index);

    scale->sigma = ui->sigmaSpinbox->value();
    scale->weight = ui->omegaSpinbox->value();

    void (QDoubleSpinBox::*signal)(double)  =
            static_cast<void (QDoubleSpinBox::*)(double)>( &QDoubleSpinBox::valueChanged );
    connect(ui->omegaSpinbox, signal, [this] (double value) { this->scale->weight = value; } );
    connect(ui->sigmaSpinbox, signal, [this] (double value) { this->scale->sigma = value; } );

    if(removeCallback != nullptr)
        connect(ui->removeButton, &QPushButton::clicked, [this, removeCallback, index, frame]() {
            removeCallback(this->index - 1);
            frame->layout()->removeWidget(this);
            delete this;
            frame->repaint();
        });

    if(frame->layout() == nullptr)
        frame->setLayout(new QVBoxLayout());
    frame->layout()->addWidget(this);

}

MultiScaleRetinexScale::~MultiScaleRetinexScale()
{
    delete ui;
}

unsigned int MultiScaleRetinexScale::getIndex() const
{
    return this->index;
}
void MultiScaleRetinexScale::setIndex(unsigned int index)
{
    this->index = index;
    this->ui->indexLabel->setText(QString::number(index));
}
