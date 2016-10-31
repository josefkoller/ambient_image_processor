#include "ImageViewControlWidget.h"
#include "ui_ImageViewControlWidget.h"

#include "ImageViewWidget.h"

ImageViewControlWidget::ImageViewControlWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::ImageViewControlWidget)
{
    ui->setupUi(this);
}

ImageViewControlWidget::~ImageViewControlWidget()
{
    delete ui;
}

void ImageViewControlWidget::on_do_rescale_checkbox_toggled(bool checked)
{
    emit this->doRescaleChanged(checked);
}

void ImageViewControlWidget::on_do_multiply_checkbox_toggled(bool checked)
{
    emit this->doMultiplyChanged(checked);
}

void ImageViewControlWidget::on_useWindowCheckbox_toggled(bool checked)
{
    emit this->useWindowChanged(checked);
}
