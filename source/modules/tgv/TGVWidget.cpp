#include "TGVWidget.h"
#include "ui_TGVWidget.h"

TGVWidget::TGVWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::TGVWidget)
{
    ui->setupUi(this);
}

TGVWidget::~TGVWidget()
{
    delete ui;
}

void TGVWidget::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(this, &BaseModuleWidget::fireWorkerFinished,
            this, [this]() {
        this->ui->stop_button->setEnabled(false);
    });
}

void TGVWidget::setIterationFinishedCallback(TGVProcessor::IterationFinished iteration_finished_callback)
{
    this->iteration_finished_callback = [this, iteration_finished_callback](uint iteration_index, uint iteration_count,
            ITKImage u){
        iteration_finished_callback(iteration_index, iteration_count, u);
        return this->stop_after_next_iteration;
    };
}

void TGVWidget::on_perform_button_clicked()
{
    this->stop_after_next_iteration = false;
    this->processInWorkerThread();
    this->ui->stop_button->setEnabled(true);
}

ITKImage TGVWidget::processImage(ITKImage image)
{
    const float alpha0 = this->ui->alpha0_spinbox->value();
    const float alpha1 = this->ui->alpha1_spinbox->value();
    const float lambda = this->ui->lambda_spinbox->value();
    const uint iteration_count = this->ui->iteration_count_spinbox->value();
    const uint paint_iteration_interval = this->ui->paint_iteration_interval_spinbox->value();

    if(this->ui->tgv1_l2_algorithm_checkbox->isChecked())
        return TGVProcessor::processTVL2GPUCuda(image, lambda,
                                                alpha0,
                                                alpha1,
                                                iteration_count,
                                            paint_iteration_interval,
                                            this->iteration_finished_callback);
    if(this->ui->tgv1_l1_algorithm_checkbox->isChecked())
        return TGVProcessor::processTVL1GPUCuda(image, lambda,
                                                alpha0,
                                                alpha1,
                                                iteration_count,
                                            paint_iteration_interval,
                                            this->iteration_finished_callback);
    if(this->ui->tgv2_l1_algorithm_checkbox->isChecked())
        return TGVProcessor::processTGV2L1GPUCuda(image, lambda,
                                                alpha0,
                                                alpha1,
                                                iteration_count,
                                            paint_iteration_interval,
                                            this->iteration_finished_callback);
    if(this->ui->tgv2_l2_algorithm_checkbox->isChecked())
        return TGVProcessor::processTGV2L2GPUCuda(image, lambda,
                                                alpha0,
                                                alpha1,
                                                iteration_count,
                                            paint_iteration_interval,
                                            this->iteration_finished_callback);
}

void TGVWidget::on_stop_button_clicked()
{
    this->stop_after_next_iteration = true;
}
