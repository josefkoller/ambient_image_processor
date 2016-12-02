#include "MorphologicalFilterWidget.h"
#include "ui_MorphologicalFilterWidget.h"

#include "CudaImageOperationsProcessor.h"

MorphologicalFilterWidget::MorphologicalFilterWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::MorphologicalFilterWidget)
{
    ui->setupUi(this);
}

MorphologicalFilterWidget::~MorphologicalFilterWidget()
{
    delete ui;
}

void MorphologicalFilterWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}

ITKImage MorphologicalFilterWidget::processImage(ITKImage image)
{
    if(this->ui->binary_dilate_checkbox->isChecked())
        return CudaImageOperationsProcessor::binary_dilate(image);

    return ITKImage();
}
