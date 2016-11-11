#include "TGVDeshadeMaskedWidget.h"
#include "ui_TGVDeshadeMaskedWidget.h"

#include "TGVDeshadeMaskedProcessor.h"

#include <QFileDialog>

TGVDeshadeMaskedWidget::TGVDeshadeMaskedWidget(QString title, QWidget* parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::TGVDeshadeMaskedWidget)
{
    ui->setupUi(this);

    this->shading_output_view = new ImageViewWidget("Denoised", this->ui->shading_frame);
    this->ui->shading_frame->layout()->addWidget(this->shading_output_view);

    this->denoised_output_view = new ImageViewWidget("Denoised", this->ui->denoised_frame);
    this->ui->denoised_frame->layout()->addWidget(this->denoised_output_view);
}

TGVDeshadeMaskedWidget::~TGVDeshadeMaskedWidget()
{
    delete ui;
}

void TGVDeshadeMaskedWidget::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(this, &BaseModuleWidget::fireWorkerFinished,
            this, [this]() {
        this->ui->stop_button->setEnabled(false);
    });

    this->shading_output_view->registerCrosshairSubmodule(image_widget);
    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this->shading_output_view, &ImageViewWidget::sliceIndexChanged);
    this->denoised_output_view->registerCrosshairSubmodule(image_widget);
    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this->denoised_output_view, &ImageViewWidget::sliceIndexChanged);

    this->mask_fetcher = MaskWidget::createMaskFetcher(image_widget);
}

void TGVDeshadeMaskedWidget::setIterationFinishedCallback(TGVDeshadeMaskedProcessor::IterationFinished iteration_finished_callback)
{
    this->iteration_finished_callback = [this, iteration_finished_callback](uint iteration_index, uint iteration_count,
            ITKImage u, ITKImage l, ITKImage r){
        iteration_finished_callback(iteration_index, iteration_count, r);

        this->shading_output_view->fireImageChange(l);
        this->denoised_output_view->fireImageChange(u);

        return this->stop_after_next_iteration;
    };
}


ITKImage TGVDeshadeMaskedWidget::processImage(ITKImage image)
{
    const float alpha0 = this->ui->alpha0_spinbox->value();
    const float alpha1 = this->ui->alpha1_spinbox->value();
    const float lambda = this->ui->lambda_spinbox->value();
    const uint iteration_count = this->ui->iteration_count_spinbox->value();
    const uint paint_iteration_interval = this->ui->paint_iteration_interval_spinbox->value();

    const bool set_negative_values_to_zero = this->ui->set_negative_values_to_zero_checkbox->isChecked();

    ITKImage mask = this->ui->use_mask_module_checkbox->isChecked() ?
        mask_fetcher() : ITKImage();

    const bool add_background_back = this->ui->add_background_back_checkbox->isChecked();

    const bool calculate_div_v = this->ui->calculate_div_v->isChecked();

    ITKImage denoised_image = ITKImage();
    ITKImage shading_image = ITKImage();
    ITKImage deshaded_image = ITKImage();
    ITKImage div_v_image = ITKImage();

    deshaded_image = TGVDeshadeMaskedProcessor::processTGV2L1GPUCuda(image, lambda,
                                              alpha0,
                                              alpha1,
                                              iteration_count,
                                              -1,
                                              paint_iteration_interval,
                                              this->iteration_finished_callback,
                                              mask,
                                              set_negative_values_to_zero,
                                              add_background_back,
                                              denoised_image,
                                              shading_image,
                                              div_v_image,
                                              calculate_div_v);

    this->denoised_output_view->setImage(denoised_image);
    this->shading_output_view->setImage(shading_image);
    return deshaded_image;
}


void TGVDeshadeMaskedWidget::on_perform_button_clicked()
{
    this->stop_after_next_iteration = false;
    this->processInWorkerThread();
    this->ui->stop_button->setEnabled(true);
}

void TGVDeshadeMaskedWidget::on_stop_button_clicked()
{
    this->stop_after_next_iteration = true;
}

void TGVDeshadeMaskedWidget::on_save_second_output_button_clicked()
{
    auto image = this->shading_output_view->getImage();
    if(image.isNull())
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save volume file");
    if(file_name.isNull())
        return;

    image.write(file_name.toStdString());
}

void TGVDeshadeMaskedWidget::on_save_denoised_button_clicked()
{
    auto image = this->denoised_output_view->getImage();
    if(image.isNull())
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save volume file");
    if(file_name.isNull())
        return;

    image.write(file_name.toStdString());
}
