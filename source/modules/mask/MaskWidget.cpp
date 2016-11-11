#include "MaskWidget.h"
#include "ui_MaskWidget.h"

#include <QFileDialog>

MaskWidget::MaskWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::MaskWidget)
{
    ui->setupUi(this);

    this->mask_view = new ImageViewWidget("Mask", this->ui->mask_frame);
    this->ui->mask_frame->layout()->addWidget(this->mask_view);
}

MaskWidget::~MaskWidget()
{
    delete ui;
}

void MaskWidget::on_load_mask_button_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "open volume file");
    if(file_name == QString::null || !QFile(file_name).exists())
        return;

    this->mask_view->setImage(ITKImage::read(file_name.toStdString()));
    this->ui->enabled_checkbox->setChecked(true);
}

ITKImage MaskWidget::getMask() const {
    if(!this->ui->enabled_checkbox->isChecked())
        return ITKImage();

    return this->mask_view->getImage();
}

void MaskWidget::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    this->mask_view->registerCrosshairSubmodule(image_widget);
    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this->mask_view, &ImageViewWidget::sliceIndexChanged);
}

MaskWidget::MaskFetcher MaskWidget::createMaskFetcher(ImageWidget *image_widget)
{
    return [image_widget]() {
        auto module = image_widget->getModuleByName("Mask");
        auto mask_module = dynamic_cast<MaskWidget*>(module);
        if(mask_module == nullptr)
            throw std::runtime_error("did not find mask module");
        return mask_module->getMask();
    };
}
