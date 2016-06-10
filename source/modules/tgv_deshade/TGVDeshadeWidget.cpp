#include "TGVDeshadeWidget.h"
#include "ui_TGVDeshadeWidget.h"

#include "TGVDeshadeProcessor.h"

#include <QFileDialog>

TGVDeshadeWidget::TGVDeshadeWidget(QString title, QWidget* parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::TGVDeshadeWidget)
{
    ui->setupUi(this);

    this->second_output_view = new ImageViewWidget("Denoised", this->ui->denoised_frame);
    this->ui->denoised_frame->layout()->addWidget(this->second_output_view);

    this->mask_view = new ImageViewWidget("Mask", this->ui->mask_frame);
    this->ui->mask_frame->layout()->addWidget(this->mask_view);
}

TGVDeshadeWidget::~TGVDeshadeWidget()
{
    delete ui;
}



void TGVDeshadeWidget::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(this, &BaseModuleWidget::fireWorkerFinished,
            this, [this]() {
        this->ui->stop_button->setEnabled(false);
    });

    this->second_output_view->registerCrosshairSubmodule(image_widget);

    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this->second_output_view, &ImageViewWidget::sliceIndexChanged);
}

void TGVDeshadeWidget::setIterationFinishedCallback(TGVDeshadeProcessor::IterationFinished iteration_finished_callback)
{
    this->iteration_finished_callback = [this, iteration_finished_callback](uint iteration_index, uint iteration_count,
            ITKImage u, ITKImage l){
        iteration_finished_callback(iteration_index, iteration_count, l);

        this->second_output_view->fireImageChange(u);
        return this->stop_after_next_iteration;
    };
}


ITKImage TGVDeshadeWidget::processImage(ITKImage image)
{
    const float alpha0 = this->ui->alpha0_spinbox->value();
    const float alpha1 = this->ui->alpha1_spinbox->value();
    const float lambda = this->ui->lambda_spinbox->value();
    const uint iteration_count = this->ui->iteration_count_spinbox->value();
    const uint paint_iteration_interval = this->ui->paint_iteration_interval_spinbox->value();

    const bool set_negative_values_to_zero = this->ui->set_negative_values_to_zero_checkbox->isChecked();
    auto mask = this->mask_view->getImage();


    return TGVDeshadeProcessor::processTGV2L1GPUCuda(image, lambda,
                                              alpha0,
                                              alpha1,
                                              iteration_count,
                                              paint_iteration_interval,
                                              this->iteration_finished_callback,
                                              mask,
                                              set_negative_values_to_zero);
}


void TGVDeshadeWidget::on_perform_button_clicked()
{
    this->stop_after_next_iteration = false;
    this->processInWorkerThread();
    this->ui->stop_button->setEnabled(true);
}

void TGVDeshadeWidget::on_stop_button_clicked()
{
    this->stop_after_next_iteration = true;
}

void TGVDeshadeWidget::on_save_second_output_button_clicked()
{
    auto image = this->second_output_view->getImage();
    if(image.isNull())
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save volume file");
    if(file_name.isNull())
        return;

    image.write(file_name.toStdString());
}

void TGVDeshadeWidget::on_load_mask_button_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "open volume file");
    if(file_name == QString::null || !QFile(file_name).exists())
        return;

    this->mask_view->setImage(ITKImage::read(file_name.toStdString()));
}