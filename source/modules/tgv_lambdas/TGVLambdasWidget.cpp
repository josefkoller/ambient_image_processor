#include "TGVLambdasWidget.h"
#include "ui_TGVLambdasWidget.h"

#include <QFileDialog>
#include "TGVLambdasProcessor.h"

TGVLambdasWidget::TGVLambdasWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::TGVLambdasWidget)
{
    ui->setupUi(this);

    this->lambdas_widget = new ImageViewWidget("Lambdas View", this->ui->lambdas_frame);
    this->ui->lambdas_frame->layout()->addWidget(this->lambdas_widget);
}

TGVLambdasWidget::~TGVLambdasWidget()
{
    delete ui;
}

void TGVLambdasWidget::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(this, &BaseModuleWidget::fireWorkerFinished,
            this, [this]() {
        this->ui->stop_button->setEnabled(false);
    });
}

void TGVLambdasWidget::on_load_button_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "open volume file");
    if(file_name == QString::null || !QFile(file_name).exists())
        return;

    this->lambdas_image = ITKImage::read(file_name.toStdString());
    this->lambdas_widget->setImage(this->lambdas_image);
}

void TGVLambdasWidget::on_perform_button_clicked()
{
    this->stop_after_next_iteration = false;
    this->processInWorkerThread();
    this->ui->stop_button->setEnabled(true);
}

ITKImage TGVLambdasWidget::processImage(ITKImage image)
{
    if(image.isNull())
        throw std::runtime_error("source image is not set");

    if(lambdas_image.isNull())
        throw std::runtime_error("lambdas image is not set");

    if(image.width != this->lambdas_image.width ||
            image.height != this->lambdas_image.height ||
            image.depth != this->lambdas_image.depth) {
        throw std::runtime_error("lambdas image does not have the same dimensions than the source image");
    }

    const float alpha0 = this->ui->alpha0_spinbox->value();
    const float alpha1 = this->ui->alpha1_spinbox->value();
    const float lambda_offset = this->ui->lambda_offset_spinbox->value();
    const float lambda_factor = this->ui->lambda_factor_spinbox->value();
    const uint iteration_count = this->ui->iteration_count_spinbox->value();
    const uint paint_iteration_interval = this->ui->paint_iteration_interval_spinbox->value();

    auto result_image = TGVLambdasProcessor::processTGV2L1LambdasGPUCuda(
                image, lambdas_image, lambda_offset, lambda_factor,
                alpha0, alpha1, iteration_count, paint_iteration_interval,
                this->iteration_finished_callback);
    return result_image;
}

void TGVLambdasWidget::setIterationFinishedCallback(TGVLambdasProcessor::IterationFinished iteration_finished_callback)
{
    this->iteration_finished_callback = [this, iteration_finished_callback](uint iteration_index, uint iteration_count,
            ITKImage u){
        iteration_finished_callback(iteration_index, iteration_count, u);
        return this->stop_after_next_iteration;
    };
}

void TGVLambdasWidget::on_stop_button_clicked()
{
    this->stop_after_next_iteration = true;
}
