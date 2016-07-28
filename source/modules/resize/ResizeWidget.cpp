#include "ResizeWidget.h"
#include "ui_ResizeWidget.h"

#include "ResizeProcessor.h"

ResizeWidget::ResizeWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::ResizeWidget)
{
    ui->setupUi(this);
}

ResizeWidget::~ResizeWidget()
{
    delete ui;
}

void ResizeWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}

ITKImage ResizeWidget::processImage(ITKImage image)
{
    ITKImage::PixelType size_factor = this->ui->size_factor_spinbox->value();
    ResizeProcessor::InterpolationMethod interpolation_method =
            this->ui->interpolate_nearest_neighbour_checkbox->isChecked() ?
                ResizeProcessor::InterpolationMethod::NearestNeighbour :
                (this->ui->interpolate_linear_checkbox->isChecked() ?
                    ResizeProcessor::InterpolationMethod::Linear :
                    ResizeProcessor::InterpolationMethod::Sinc);

    return ResizeProcessor::process(image, size_factor, interpolation_method);
}
