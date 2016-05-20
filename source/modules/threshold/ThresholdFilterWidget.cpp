#include "ThresholdFilterWidget.h"
#include "ui_ThresholdFilterWidget.h"

#include "ThresholdFilterProcessor.h"

ThresholdFilterWidget::ThresholdFilterWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::ThresholdFilterWidget)
{
    ui->setupUi(this);
}

ThresholdFilterWidget::~ThresholdFilterWidget()
{
    delete ui;
}

ITKImage ThresholdFilterWidget::processImage(ITKImage image)
{
    if(image.isNull())
        return ITKImage();

    auto lower_threshold_value = this->ui->lowerThresholdSpinbox->value();
    auto upper_threshold_value = this->ui->upperThresholdSpinbox->value();
    auto outside_pixel_value = this->ui->outsideSpinbox->value();

    return ThresholdFilterProcessor::process( image,
                                              lower_threshold_value,
                                              upper_threshold_value,
                                              outside_pixel_value);
}

void ThresholdFilterWidget::on_thresholdButton_clicked()
{
    this->processInWorkerThread();
}
