#include "RescaleIntensityWidget.h"
#include "ui_RescaleIntensityWidget.h"

#include "RescaleIntensityProcessor.h"

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
    return RescaleIntensityProcessor::process(image,
                                              this->ui->minimum_spinbox->value(),
                                              this->ui->maximum_spinbox->value());
}
