#include "BSplineInterpolationWidget.h"
#include "ui_BSplineInterpolationWidget.h"

#include "BSplineInterpolationProcessor.h"

#include <QFileDialog>

BSplineInterpolationWidget::BSplineInterpolationWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::BSplineInterpolationWidget)
{
    ui->setupUi(this);
}

BSplineInterpolationWidget::~BSplineInterpolationWidget()
{
    delete ui;
}


void BSplineInterpolationWidget::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    this->mask_fetcher = MaskWidget::createMaskFetcher(image_widget);
}

void BSplineInterpolationWidget::on_performButton_clicked()
{
    this->processInWorkerThread();
}

ITKImage BSplineInterpolationWidget::processImage(ITKImage image) {
    uint spline_order = this->ui->splineOrderSpinbox->value();
    uint number_of_fitting_levels = this->ui->numberOfFittingLevelsSpinbox->value();
    uint number_of_nodes = this->ui->numberOfNodesSpinbox->value();

    if(number_of_nodes <= spline_order)
        throw std::runtime_error("Number of nodes must be greater than the spline order");

    ITKImage mask = this->ui->use_mask_module_checkbox->isChecked() ?
        mask_fetcher() : ITKImage();

    return BSplineInterpolationProcessor::process(image,
      mask,
      spline_order, number_of_nodes, number_of_fitting_levels);
}
