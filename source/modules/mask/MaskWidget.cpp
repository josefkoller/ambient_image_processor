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

#include "MaskWidget.h"
#include "ui_MaskWidget.h"

#include <QFileDialog>
#include <QMessageBox>

MaskWidget::MaskWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::MaskWidget)
{
    ui->setupUi(this);

    this->mask_view = new ImageViewWidget("Mask", this->ui->mask_frame);
    this->ui->mask_frame->layout()->addWidget(this->mask_view);

    connect(this->mask_view, &ImageViewWidget::imageChanged,
            this, &MaskWidget::maskChanged);
}

MaskWidget::~MaskWidget()
{
    delete ui;
}

void MaskWidget::on_load_mask_button_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "open volume file");
    if(file_name == QString::null || !QFile(file_name).exists())
        return;

    auto mask = ITKImage::read(file_name.toStdString());
    this->mask_view->setImage(mask);

    if(!mask.hasSameSize(this->image)) {
        this->ui->enabled_checkbox->setChecked(false);

        this->setStatusText("mask module deactivated, mask and image have different size");
        QMessageBox::information(this, "Info", "mask module deactivated, mask and image have different size");
        return;
    }

    this->ui->enabled_checkbox->setChecked(true);
}

ITKImage MaskWidget::getMask() {
    if(!this->ui->enabled_checkbox->isChecked())
        return ITKImage();

    auto mask = this->mask_view->getImage();
    if(!mask.hasSameSize(this->image)) {
        this->setStatusText("mask module deactivated, mask and image have different size");
        return ITKImage();
    }

    return mask;
}

void MaskWidget::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    this->mask_view->registerCrosshairSubmodule(image_widget);
    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this->mask_view, &ImageViewWidget::sliceIndexChanged);

    // disable mask, if it has the wrong dimensions...
    connect(image_widget, &ImageWidget::imageChanged,
            this, [this](ITKImage image) {
        this->image = image;
        auto mask = this->getMask();
        if(!mask.hasSameSize(image)) {
            this->ui->enabled_checkbox->setChecked(false);
            this->setStatusText("mask module deactivated, mask and image have different size");
        }
    });
}

MaskWidget::MaskFetcher MaskWidget::createMaskFetcher(ImageWidget *image_widget)
{
    return [image_widget]() {
        auto module = image_widget->getModuleByName("Mask");
        auto mask_module = dynamic_cast<MaskWidget*>(module);
        if(mask_module == nullptr)
            throw std::runtime_error("did not find mask module");
        return mask_module->getMask();
    };
}

void MaskWidget::setMaskImage(ITKImage mask)
{
    this->ui->enabled_checkbox->setChecked(true);
    this->mask_view->setImage(mask);
}

void MaskWidget::on_enabled_checkbox_clicked()
{
    emit this->maskChanged(this->getMask());
}
