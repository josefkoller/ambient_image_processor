#include "ImageInformationWidget.h"
#include "ui_ImageInformationWidget.h"

#include "ImageWidget.h"
#include "ImageInformationProcessor.h"

ImageInformationWidget::ImageInformationWidget(QWidget *parent) :
    BaseModuleWidget(parent),
    ui(new Ui::ImageInformationWidget)
{
    ui->setupUi(this);
}

ImageInformationWidget::~ImageInformationWidget()
{
    delete ui;
}

void ImageInformationWidget::collectInformation(ITKImage::InnerITKImage::Pointer image)
{
    auto information = ImageInformationProcessor::collectInformation(image);

    this->ui->dimensions_label->setText(information["dimensions"]);
    this->ui->voxel_count_label->setText(information["voxel_count"]);
    this->ui->mean_label->setText(information["mean"]);
    this->ui->standard_deviation_label->setText(information["standard_deviation"]);
    this->ui->variance_label->setText(information["variance"]);
    this->ui->standard_error_label->setText(information["standard_error"]);
    this->ui->minimum_label->setText(information["minimum"]);
    this->ui->maximum_label->setText(information["maximum"]);
    this->ui->origin_label->setText(information["origin"]);
    this->ui->spacing_label->setText(information["spacing"]);
}

void ImageInformationWidget::registerModule(ImageWidget* image_widget)
{
    connect(image_widget, &ImageWidget::imageChanged,
            this, &ImageInformationWidget::collectInformation);
}
