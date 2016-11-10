#include "NormalizeStatisticWidget.h"
#include "ui_NormalizeStatisticWidget.h"

#include <QFileDialog>

#include "NormalizeStatisticProcessor.h"

NormalizeStatisticWidget::NormalizeStatisticWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::NormalizeStatisticWidget)
{
    ui->setupUi(this);

    this->reference_image_widget = new ImageViewWidget("Reference Image View", this->ui->second_image_frame);
    this->ui->second_image_frame->layout()->addWidget(this->reference_image_widget);
}

NormalizeStatisticWidget::~NormalizeStatisticWidget()
{
    delete ui;
}

void NormalizeStatisticWidget::on_load_button_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "open volume file");
    if(file_name == QString::null || !QFile(file_name).exists())
        return;

    this->reference_image_widget->setImage(ITKImage::read(file_name.toStdString()));
}

void NormalizeStatisticWidget::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this->reference_image_widget, &ImageViewWidget::sliceIndexChanged);

    this->reference_image_widget->registerCrosshairSubmodule(image_widget);
}

ITKImage NormalizeStatisticWidget::processImage(ITKImage image)
{
    auto reference_image = this->reference_image_widget->getImage();
    if(reference_image.isNull())
        throw std::runtime_error("No reference image given");

    if(this->ui->mean_add_constant_checkbox->isChecked())
        return NormalizeStatisticProcessor::equalizeMeanAddConstant(image, reference_image);
    else if(this->ui->mean_factor_checkbox->isChecked())
        return NormalizeStatisticProcessor::equalizeMeanScale(image, reference_image);
    else if(this->ui->maxmin_checkbox->isChecked())
            return NormalizeStatisticProcessor::equalizeMaxMin(image, reference_image);

    return NormalizeStatisticProcessor::equalizeStandardDeviation(image, reference_image);
}

void NormalizeStatisticWidget::on_performButton_clicked()
{
    this->processInWorkerThread();
}
