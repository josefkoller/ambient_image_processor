#include "TGVWidget.h"
#include "ui_TGVWidget.h"

TGVWidget::TGVWidget(QWidget *parent) :
    BaseModuleWidget(parent),
    ui(new Ui::TGVWidget)
{
    ui->setupUi(this);
}

TGVWidget::~TGVWidget()
{
    delete ui;
}

void TGVWidget::setIterationFinishedCallback(TGVProcessor::IterationFinished iteration_finished_callback)
{
    this->iteration_finished_callback = iteration_finished_callback;
}

void TGVWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}

ITKImage TGVWidget::processImage(ITKImage image)
{
    const float alpha0 = this->ui->alpha0_spinbox->value();
    const float alpha1 = this->ui->alpha1_spinbox->value();
    const float lambda = this->ui->lambda_spinbox->value();
    const uint iteration_count = this->ui->iteration_count_spinbox->value();

    bool perform_on_gpu = this->ui->gpu_radio_button->isChecked();

    if(perform_on_gpu)
        return TGVProcessor::processTVL2GPU(image, lambda, iteration_count, this->iteration_finished_callback);

    return TGVProcessor::processTVL2CPU(image, lambda, iteration_count, this->iteration_finished_callback);
}
