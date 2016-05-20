#include "BilateralFilterWidget.h"
#include "ui_BilateralFilterWidget.h"

#include "BilateralFilterProcessor.h"

BilateralFilterWidget::BilateralFilterWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::BilateralFilterWidget)
{
    ui->setupUi(this);
}

BilateralFilterWidget::~BilateralFilterWidget()
{
    delete ui;
}

void BilateralFilterWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}

ITKImage BilateralFilterWidget::processImage(ITKImage image)
{
    if(image.isNull())
        return ITKImage();

    float sigma_spatial_distance = this->ui->sigmaSpatialDistanceSpinbox->value();
    float sigma_intensity_distance = this->ui->sigmaIntensityDistanceSpinbox->value();
    int kernel_size = this->ui->kernelSizeSpinbox->value();

    return BilateralFilterProcessor::process( image,
                                                sigma_spatial_distance,
                                                sigma_intensity_distance,
                                                kernel_size);
}
