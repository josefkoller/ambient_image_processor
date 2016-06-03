#include "RescaleIntensityWidget.h"
#include "ui_RescaleIntensityWidget.h"

#include <itkRescaleIntensityImageFilter.h>

RescaleIntensityWidget::RescaleIntensityWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::RescaleIntensityWidget)
{
    ui->setupUi(this);
}

RescaleIntensityWidget::~RescaleIntensityWidget()
{
    delete ui;
}

void RescaleIntensityWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}

ITKImage RescaleIntensityWidget::processImage(ITKImage image)
{
    typedef ITKImage::InnerITKImage Image;
    typedef itk::RescaleIntensityImageFilter<Image, Image> RescaleFilter;
    RescaleFilter::Pointer rescale_filter = RescaleFilter::New();
    rescale_filter->SetInput(image.getPointer());
    rescale_filter->SetOutputMinimum(this->ui->minimum_spinbox->value());
    rescale_filter->SetOutputMaximum(this->ui->maximum_spinbox->value());
    rescale_filter->Update();

    Image::Pointer result = rescale_filter->GetOutput();
    result->DisconnectPipeline();

    return ITKImage(result);
}
