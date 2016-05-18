#include "MultiScaleRetinexWidget.h"
#include "ui_MultiScaleRetinexWidget.h"

#include "MultiScaleRetinexProcessor.h"

MultiScaleRetinexWidget::MultiScaleRetinexWidget(QWidget *parent) :
    BaseModuleWidget(parent),
    ui(new Ui::MultiScaleRetinexWidget)
{
    ui->setupUi(this);
}

MultiScaleRetinexWidget::~MultiScaleRetinexWidget()
{
    delete ui;
}

void MultiScaleRetinexWidget::on_addScaleButton_clicked()
{
    this->multi_scale_retinex.addScaleTo(this->ui->multiScaleRetinexScalesFrame);
}

void MultiScaleRetinexWidget::on_calculate_button_clicked()
{
   this->processInWorkerThread();
}

ITKImage MultiScaleRetinexWidget::processImage(ITKImage image)
{
    return MultiScaleRetinexProcessor::process(image,
         this->multi_scale_retinex.scales);
}
