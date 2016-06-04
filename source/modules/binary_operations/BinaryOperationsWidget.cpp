#include "BinaryOperationsWidget.h"
#include "ui_BinaryOperationsWidget.h"

#include <QFileDialog>

#include "CudaImageOperationsProcessor.h"

BinaryOperationsWidget::BinaryOperationsWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::BinaryOperationsWidget)
{
    ui->setupUi(this);

    this->second_image_widget = new ImageViewWidget("Second Image View", this->ui->second_image_frame);
    this->ui->second_image_frame->layout()->addWidget(this->second_image_widget);
}

BinaryOperationsWidget::~BinaryOperationsWidget()
{
    delete ui;
}

void BinaryOperationsWidget::on_load_button_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "open volume file");
    if(file_name == QString::null || !QFile(file_name).exists())
        return;

    this->second_image_widget->setImage(ITKImage::read(file_name.toStdString()));
}

void BinaryOperationsWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}

ITKImage BinaryOperationsWidget::processImage(ITKImage image1)
{
    auto image2 = this->second_image_widget->getImage();

    if(this->ui->divide_checkbox->isChecked())
        return CudaImageOperationsProcessor::divide(image1, image2);
    if(this->ui->multiply_checkbox->isChecked())
        return CudaImageOperationsProcessor::multiply(image1, image2);
    if(this->ui->add_checkbox->isChecked())
        return CudaImageOperationsProcessor::add(image1, image2);


    return ITKImage();
}

void BinaryOperationsWidget::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this->second_image_widget, &ImageViewWidget::sliceIndexChanged);

    this->second_image_widget->registerCrosshairSubmodule(image_widget);
}
