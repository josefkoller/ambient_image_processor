#include "TGVNonParametricDeshadeWidget.h"
#include "ui_TGVNonParametricDeshadeWidget.h"

#include "TGVDeshadeProcessor.h"
#include "TGVNonParametricDeshadeProcessor.h"

#include <QFileDialog>

TGVNonParametricDeshadeWidget::TGVNonParametricDeshadeWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::TGVNonParametricDeshadeWidget)
{
    ui->setupUi(this);

    this->shading_output_view = new ImageViewWidget("Denoised", this->ui->shading_frame);
    this->ui->shading_frame->layout()->addWidget(this->shading_output_view);

    this->denoised_output_view = new ImageViewWidget("Denoised", this->ui->denoised_frame);
    this->ui->denoised_frame->layout()->addWidget(this->denoised_output_view);

    this->mask_view = new ImageViewWidget("Mask", this->ui->mask_frame);
    this->ui->mask_frame->layout()->addWidget(this->mask_view);
}

TGVNonParametricDeshadeWidget::~TGVNonParametricDeshadeWidget()
{
    delete ui;
}

void TGVNonParametricDeshadeWidget::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    this->shading_output_view->registerCrosshairSubmodule(image_widget);
    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this->shading_output_view, &ImageViewWidget::sliceIndexChanged);
    this->denoised_output_view->registerCrosshairSubmodule(image_widget);
    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this->denoised_output_view, &ImageViewWidget::sliceIndexChanged);
}

ITKImage TGVNonParametricDeshadeWidget::processImage(ITKImage image)
{
    const float lambda = this->ui->lambda_spinbox->value();
    auto mask = this->mask_view->getImage();

    const uint check_iteration_count = this->ui->check_iteration_count_spinbox->value();
    const ITKImage::PixelType alpha_step_minimum = this->ui->alpha_step_minimum_spinbox->value();
    const uint final_iteration_count = this->ui->final_iteration_count_spinbox->value();

    ITKImage denoised_image = ITKImage();
    ITKImage shading_image = ITKImage();
    ITKImage deshaded_image = ITKImage();

  /*
        // optimization by calling the standard tgv deshade cuda routine
        TGVDeshadeProcessor::processTGV2L1DeshadeCuda_convergenceOptimization(
                    image,
                    lambda,
                    mask,
                    false,
                    denoised_image,
                    shading_image,
                    deshaded_image);
    */
        TGVNonParametricDeshadeProcessor::performTGVDeshade( // special tgv cuda deshade routine
                    image, lambda, mask,
                    check_iteration_count, alpha_step_minimum, final_iteration_count,
                    denoised_image, shading_image, deshaded_image);


    this->denoised_output_view->setImage(denoised_image);
    this->shading_output_view->setImage(shading_image);
    return deshaded_image;
}

void TGVNonParametricDeshadeWidget::on_load_mask_button_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "open volume file");
    if(file_name == QString::null || !QFile(file_name).exists())
        return;

    this->mask_view->setImage(ITKImage::read(file_name.toStdString()));
}

void TGVNonParametricDeshadeWidget::on_save_denoised_button_clicked()
{
    auto image = this->denoised_output_view->getImage();
    if(image.isNull())
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save volume file");
    if(file_name.isNull())
        return;

    image.write(file_name.toStdString());
}

void TGVNonParametricDeshadeWidget::on_save_second_output_button_clicked()
{
    auto image = this->shading_output_view->getImage();
    if(image.isNull())
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save volume file");
    if(file_name.isNull())
        return;

    image.write(file_name.toStdString());
}

void TGVNonParametricDeshadeWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}
