#include "ShrinkWidget.h"
#include "ui_ShrinkWidget.h"

#include "ShrinkProcessor.h"

ShrinkWidget::ShrinkWidget(QWidget *parent) :
    BaseModuleWidget(parent),
    ui(new Ui::ShrinkWidget)
{
    ui->setupUi(this);
}

ShrinkWidget::~ShrinkWidget()
{
    delete ui;
}

void ShrinkWidget::on_shrink_button_clicked()
{
    this->processInWorkerThread();
}

ITKImage ShrinkWidget::processImage(ITKImage image)
{
    if(image.isNull())
        return ITKImage();

    unsigned int shrink_factor_x = this->ui->shrink_factor_x->text().toUInt();
    unsigned int shrink_factor_y = this->ui->shrink_factor_y->text().toUInt();
    unsigned int shrink_factor_z = this->ui->shrink_factor_z->text().toUInt();

    return ShrinkProcessor::process(image, shrink_factor_x, shrink_factor_y, shrink_factor_z);
}

void ShrinkWidget::registerModule(ImageWidget* image_widget)
{
    BaseModuleWidget::registerModule(image_widget);
}
