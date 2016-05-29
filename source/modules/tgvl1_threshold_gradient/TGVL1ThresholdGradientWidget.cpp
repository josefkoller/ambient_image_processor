#include "TGVL1ThresholdGradientWidget.h"
#include "ui_TGVL1ThresholdGradientWidget.h"

#include "ImageWidget.h"

#include "TGVL1ThresholdGradientProcessor.h"

#include "CudaImageOperationsProcessor.h"

TGVL1ThresholdGradientWidget::TGVL1ThresholdGradientWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::TGVL1ThresholdGradientWidget)
{
    ui->setupUi(this);
}

TGVL1ThresholdGradientWidget::~TGVL1ThresholdGradientWidget()
{
    delete ui;
}

void TGVL1ThresholdGradientWidget::registerModule(ImageWidget *module_container)
{
    BaseModuleWidget::registerModule(module_container);
    image_widget = module_container;


}

void TGVL1ThresholdGradientWidget::setResult(QString submodule, ITKImage image)
{
    image_widget->getOutputWidget()->setImage(image);
    setStatusText("result of " + submodule);
}

void TGVL1ThresholdGradientWidget::on_gradient_magnitude_button_clicked()
{
    auto gradient_image = TGVL1ThresholdGradientProcessor::gradient_magnitude(
                this->getSourceImage());
    this->setResult("Gradient Magnitude", gradient_image);

    result_images["Gradient Magnitude"] = gradient_image;
}

ITKImage TGVL1ThresholdGradientWidget::processImage(ITKImage source_image)
{
    auto gradient_image = result_images["Gradient Magnitude"];
    if(gradient_image.isNull())  {
        gradient_image = TGVL1ThresholdGradientProcessor::gradient_magnitude(
                    source_image);
        result_images["Gradient Magnitude"] = gradient_image;
    }

    auto threshold_value = this->ui->gradient_magnitude_theshold_spinbox->value();

    auto mask_image = TGVL1ThresholdGradientProcessor::threshold_upper_to_zero(
                gradient_image, threshold_value);
    result_images["Mask"] = mask_image;
    result_images["Thresholded Gradient Magnitude"] = CudaImageOperationsProcessor::multiply(
                gradient_image, mask_image);

    ITKImage gradient_x, gradient_y, gradient_z;
    TGVL1ThresholdGradientProcessor::gradient(
                source_image, gradient_x, gradient_y, gradient_z);

    result_images["gradient_x"] = gradient_x;
    result_images["gradient_y"] = gradient_y;
    result_images["gradient_z"] = gradient_z;

    result_images["Thresholded gradient_x"] = CudaImageOperationsProcessor::multiply(
                gradient_x, mask_image);
    result_images["Thresholded gradient_y"] = CudaImageOperationsProcessor::multiply(
                gradient_y, mask_image);

    if(source_image.depth > 1)
        result_images["Thresholded gradient_z"] = CudaImageOperationsProcessor::multiply(
                gradient_z, mask_image);

    auto iteration_callback = [this](uint iteration_index, uint iteration_count,
            ITKImage u) {
        this->setResult(QString("TGVL1 Thresholded Gradient, iteration %0/%1").arg(
                            QString::number(iteration_index),
                            QString::number(iteration_count)), u);
    };

    result_images["Shading"] = TGVL1ThresholdGradientProcessor::tgv2_l1_threshold_gradient(
                source_image,
                result_images["Thresholded gradient_x"],
            result_images["Thresholded gradient_y"],
            result_images["Thresholded gradient_z"],
            this->ui->lambda_spinbox->value(),
            this->ui->iteration_count_spinbox->value(),
            this->ui->paint_interval_spinbox->value(),
            iteration_callback,
            2,1); // alpha0, alpha1

    return result_images["Shading"];
}

void TGVL1ThresholdGradientWidget::on_thresholded_gradient_magnitude_button_clicked()
{
    this->processInWorkerThread();
}
