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

    connect(this->mask_view, &ImageViewWidget::imageChanged,
            this, &MaskWidget::maskChanged);
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

    this->ui->enabled_checkbox->setChecked(true);
    this->mask_view->setImage(ITKImage::read(file_name.toStdString()));
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

    // disable mask, if it has the wrong dimensions...
    connect(image_widget, &ImageWidget::imageChanged,
            this, [this](ITKImage image) {
        auto mask = this->getMask();
        if(!mask.hasSameSize(image))
            this->ui->enabled_checkbox->setChecked(false);
    });
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

void MaskWidget::setMaskImage(ITKImage mask)
{
    this->ui->enabled_checkbox->setChecked(true);
    this->mask_view->setImage(mask);
}

void MaskWidget::on_enabled_checkbox_clicked()
{
    emit this->maskChanged(this->getMask());
}
