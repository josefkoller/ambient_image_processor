#include "NonLocalGradientWidget.h"
#include "ui_NonLocalGradientWidget.h"

NonLocalGradientWidget::NonLocalGradientWidget(QWidget *parent) :
    BaseModuleWidget(parent),
    ui(new Ui::NonLocalGradientWidget)
{
    ui->setupUi(this);
}

NonLocalGradientWidget::~NonLocalGradientWidget()
{
    delete ui;
}

void NonLocalGradientWidget::on_kernel_size_spinbox_valueChanged(int arg1)
{
    auto kernel_size = this->ui->kernel_size_spinbox->value();
    this->ui->kernel_view->setKernelSize(kernel_size);
}

void NonLocalGradientWidget::on_sigma_spinbox_valueChanged(double arg1)
{
    auto sigma = this->ui->sigma_spinbox->value();
    this->ui->kernel_view->setSigma(sigma);
}

void NonLocalGradientWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}

ITKImage NonLocalGradientWidget::processImage(ITKImage image)
{
    uint kernel_size = this->ui->kernel_size_spinbox->value();
    float kernel_sigma = this->ui->sigma_spinbox->value();

    ITKImage result_image = NonLocalGradientProcessor::process(
                image, kernel_size, kernel_sigma);

    return result_image;
}

float NonLocalGradientWidget::getKernelSigma() const
{
    return this->ui->sigma_spinbox->value();
}
uint NonLocalGradientWidget::getKernelSize() const
{
    return this->ui->kernel_size_spinbox->value();
}
