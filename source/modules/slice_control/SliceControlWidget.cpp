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

#include "SliceControlWidget.h"
#include "ui_SliceControlWidget.h"

SliceControlWidget::SliceControlWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::SliceControlWidget),
    image(ITKImage::Null),
    visible_slice_index(0)
{
    ui->setupUi(this);
}

SliceControlWidget::~SliceControlWidget()
{
    delete ui;
}

void SliceControlWidget::registerModule(ImageWidget *image_widget)
{
    connect(image_widget, &ImageWidget::imageChanged,
            this, [this](ITKImage image){
        this->image = image;
        this->setInputRanges();

        auto container = dynamic_cast<QWidget*>(this->parent());
        container->setVisible(this->image.depth > 1);
    });

    connect(this, &SliceControlWidget::sliceIndexChanged,
            image_widget, &ImageWidget::sliceIndexChanged);

    connect(image_widget, &ImageWidget::mouseWheelOnImage,
            this, &SliceControlWidget::mouseWheelOnImage);
}

void SliceControlWidget::on_slice_spinbox_valueChanged(int user_slice_index)
{
    if(user_slice_index != this->visible_slice_index)
    {
        this->setSliceIndex(user_slice_index);
    }
}

void SliceControlWidget::on_slice_slider_valueChanged(int user_slice_index)
{
    if(user_slice_index != this->visible_slice_index)
    {
        this->setSliceIndex(user_slice_index);
    }
}

uint SliceControlWidget::userSliceIndex() const
{
    return this->ui->slice_slider->value();
}

void SliceControlWidget::setSliceIndex(uint slice_index)
{
    if(this->visible_slice_index == slice_index)
        return;

    this->visible_slice_index = slice_index;

    if(this->ui->slice_slider->value() != slice_index)
        this->ui->slice_slider->setValue(slice_index);

    if(this->ui->slice_spinbox->value() != slice_index)
        this->ui->slice_spinbox->setValue(slice_index);

    emit this->sliceIndexChanged(slice_index);
}

void SliceControlWidget::connectTo(BaseModule *other)
{
    auto other_module = dynamic_cast<SliceControlWidget*>(other);
    if(other_module == nullptr)
        return;

    connect(other_module, &SliceControlWidget::sliceIndexChanged,
            this, &SliceControlWidget::connectedSliceControlChanged);
}

void SliceControlWidget::connectedSliceControlChanged(uint slice_index)
{
    this->setSliceIndex(slice_index);
}

void SliceControlWidget::setInputRanges()
{
    if(this->image.isNull())
        return;

    if(visible_slice_index >= this->image.depth)
        this->setSliceIndex(0);

    this->ui->slice_slider->setMinimum(0); // first slice gets slice index 0
    this->ui->slice_slider->setMaximum(this->image.depth - 1);

    this->ui->slice_spinbox->setMinimum(this->ui->slice_slider->minimum());
    this->ui->slice_spinbox->setMaximum(this->ui->slice_slider->maximum());
}

void SliceControlWidget::mouseWheelOnImage(int delta)
{
    delta /= 8 * 15;
    this->setSliceIndex(this->visible_slice_index + delta);
}
