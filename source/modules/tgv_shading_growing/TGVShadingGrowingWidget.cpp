#include "TGVShadingGrowingWidget.h"
#include "ui_TGVShadingGrowingWidget.h"

#include "TGVShadingGrowingProcessor.h"

TGVShadingGrowingWidget::TGVShadingGrowingWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::TGVShadingGrowingWidget)
{
    ui->setupUi(this);
}

TGVShadingGrowingWidget::~TGVShadingGrowingWidget()
{
    delete ui;
}

void TGVShadingGrowingWidget::on_perform_button_clicked()
{
    this->stop_after_next_iteration = false;
    this->processInWorkerThread();
    this->ui->stop_button->setEnabled(true);
}

ITKImage TGVShadingGrowingWidget::processImage(ITKImage image)
{
    const float alpha0 = this->ui->alpha0_spinbox->value();
    const float alpha1 = this->ui->alpha1_spinbox->value();
    const float lambda = this->ui->lambda_spinbox->value();
    const uint iteration_count = this->ui->iteration_count_spinbox->value();
    const uint paint_iteration_interval = this->ui->paint_iteration_interval_spinbox->value();

    const float lower_threshold = this->ui->lower_threshold_spinbox->value();
    const float upper_threshold = this->ui->upper_threshold_spinbox->value();

    const uint kernel_size = this->ui->kernel_size_spinbox->value();
    const float kernel_sigma = this->ui->kernel_sigma_spinbox->value();

    return TGVShadingGrowingProcessor::process(
                image, lambda,
                alpha0,
                alpha1,
                iteration_count,
                paint_iteration_interval,
                this->iteration_finished_callback,
                lower_threshold,
                upper_threshold,
                kernel_sigma,
                kernel_size);
}

void TGVShadingGrowingWidget::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(this, &BaseModuleWidget::fireWorkerFinished,
            this, [this]() {
        this->ui->stop_button->setEnabled(false);
    });
}

void TGVShadingGrowingWidget::on_stop_button_clicked()
{
    this->stop_after_next_iteration = true;
}

void TGVShadingGrowingWidget::setIterationFinishedCallback(IterationFinished iteration_finished_callback)
{
    this->iteration_finished_callback = [this, iteration_finished_callback](uint iteration_index, uint iteration_count,
            ITKImage u){
        iteration_finished_callback(iteration_index, iteration_count, u);
        return this->stop_after_next_iteration;
    };
}
