#include "UnsharpMaskingWidget.h"
#include "ui_UnsharpMaskingWidget.h"

#include "UnsharpMaskingProcessor.h"

UnsharpMaskingWidget::UnsharpMaskingWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::UnsharpMaskingWidget)
{
    ui->setupUi(this);
}

UnsharpMaskingWidget::~UnsharpMaskingWidget()
{
    delete ui;
}

void UnsharpMaskingWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}

ITKImage UnsharpMaskingWidget::processImage(ITKImage image)
{
    auto kernel_size = this->ui->kernel_size->value();
    auto kernel_sigma = this->ui->kernel_sigma->value();
    auto factor = this->ui->factor->value();

    return UnsharpMaskingProcessor::process(image, kernel_size, kernel_sigma, factor);
}

void UnsharpMaskingWidget::on_factor_valueChanged(double arg1)
{
    this->processInWorkerThread();
}

void UnsharpMaskingWidget::on_kernel_sigma_valueChanged(double arg1)
{
    this->processInWorkerThread();
}

void UnsharpMaskingWidget::on_kernel_size_valueChanged(int arg1)
{
    this->processInWorkerThread();
}
