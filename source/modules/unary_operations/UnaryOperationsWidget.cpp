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

    if(this->ui->dct_checkbox->isChecked())
        return CudaImageOperationsProcessor::cosineTransform(image);
    if(this->ui->idct_checkbox->isChecked())
        return CudaImageOperationsProcessor::inverseCosineTransform(image);
}

void UnaryOperationsWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}
