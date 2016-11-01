#include "BSplineInterpolationWidget.h"
#include "ui_BSplineInterpolationWidget.h"

#include "BSplineInterpolationProcessor.h"

#include <QFileDialog>

BSplineInterpolationWidget::BSplineInterpolationWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::BSplineInterpolationWidget)
{
    ui->setupUi(this);

    this->mask_view = new ImageViewWidget("Mask", this->ui->mask_frame);
    this->ui->mask_frame->layout()->addWidget(this->mask_view);
}

BSplineInterpolationWidget::~BSplineInterpolationWidget()
{
    delete ui;
}


void BSplineInterpolationWidget::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    this->mask_view->registerCrosshairSubmodule(image_widget);
    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this->mask_view, &ImageViewWidget::sliceIndexChanged);
}

void BSplineInterpolationWidget::on_performButton_clicked()
{
    this->processInWorkerThread();
}

void BSplineInterpolationWidget::on_load_mask_button_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "open volume file");
    if(file_name == QString::null || !QFile(file_name).exists())
        return;

    this->mask_view->setImage(ITKImage::read(file_name.toStdString()));
}

void BSplineInterpolationWidget::on_clear_mask_button_clicked()
{
    this->mask_view->setImage(ITKImage());
}

ITKImage BSplineInterpolationWidget::processImage(ITKImage image) {
    uint spline_order = this->ui->splineOrderSpinbox->value();
    uint number_of_fitting_levels = this->ui->numberOfFittingLevelsSpinbox->value();

    return BSplineInterpolationProcessor::process(image,
      this->mask_view->getImage(),
      spline_order, number_of_fitting_levels);
}
