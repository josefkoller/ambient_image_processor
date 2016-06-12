#include "TGVNonParametricDeshade.h"
#include "ui_TGVDeshadeWidget.h"

TGVNonParametricDeshade::TGVNonParametricDeshade(QString title, QWidget *parent) :
    TGVDeshadeWidget(title, parent)
{
    ui->iteration_count_box->setVisible(false);
    ui->alpha0_box->setVisible(false);
    ui->alpha1_box->setVisible(false);
    ui->paint_interval_box->setVisible(false);
    ui->stop_button->setVisible(false);
}

TGVNonParametricDeshade::~TGVNonParametricDeshade()
{
    delete ui;
}

ITKImage TGVNonParametricDeshade::processImage(ITKImage image)
{
    const float lambda = this->ui->lambda_spinbox->value();
    const bool set_negative_values_to_zero = this->ui->set_negative_values_to_zero_checkbox->isChecked();
    auto mask = this->mask_view->getImage();

    ITKImage denoised_image = ITKImage();
    ITKImage shading_image = ITKImage();
    ITKImage deshaded_image = ITKImage();
    TGVDeshadeProcessor::processTGV2L1DeshadeCuda_convergenceOptimization(
                image,
                lambda,
                mask,
                set_negative_values_to_zero,
                denoised_image,
                shading_image,
                deshaded_image);
    this->denoised_output_view->setImage(denoised_image);
    this->shading_output_view->setImage(shading_image);
    return deshaded_image;
}
