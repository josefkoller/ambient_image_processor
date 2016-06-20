#include "UnaryOperationsWidget.h"
#include "ui_UnaryOperationsWidget.h"

#include "CudaImageOperationsProcessor.h"

UnaryOperationsWidget::UnaryOperationsWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::UnaryOperationsWidget)
{
    ui->setupUi(this);
}

UnaryOperationsWidget::~UnaryOperationsWidget()
{
    delete ui;
}

ITKImage UnaryOperationsWidget::processImage(ITKImage image)
{
    if(this->ui->invert_checkbox->isChecked())
        return CudaImageOperationsProcessor::invert(image);

    if(this->ui->binarize_checkbox->isChecked())
        return CudaImageOperationsProcessor::binarize(image);

    if(this->ui->dct_checkbox->isChecked())
        return CudaImageOperationsProcessor::cosineTransform(image);
    if(this->ui->idct_checkbox->isChecked())
        return CudaImageOperationsProcessor::inverseCosineTransform(image);

    if(this->ui->exp_checkbox->isChecked())
        return CudaImageOperationsProcessor::log(image);
    if(this->ui->log_checkbox->isChecked())
        return CudaImageOperationsProcessor::exp(image);
}

void UnaryOperationsWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}
