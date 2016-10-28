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
    uint width = this->ui->resized_width->value();
    uint height = this->ui->resized_height->value();
    uint depth = this->ui->resized_depth->value();

    if(this->ui->interpolate_auto->isChecked())
    {
        return ResizeProcessor::process(image, width, height, depth);
    }

    ResizeProcessor::InterpolationMethod interpolation_method =
            this->ui->interpolate_nearest_neighbour_checkbox->isChecked() ?
                ResizeProcessor::InterpolationMethod::NearestNeighbour :
                (this->ui->interpolate_linear_checkbox->isChecked() ?
                    ResizeProcessor::InterpolationMethod::Linear :
                    (this->ui->interpolate_bspline3_checkbox->isChecked() ?
                     ResizeProcessor::InterpolationMethod::BSpline3 :
                     ResizeProcessor::InterpolationMethod::Sinc));

    return ResizeProcessor::process(image, width, height, depth, interpolation_method);
}
