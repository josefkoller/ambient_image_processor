#include "TGVKDeshadeDownsampledWidget.h"
#include "ui_TGVKDeshadeDownsampledWidget.h"

#include <QFileDialog>

#include "TGVKDeshadeProcessor.h"

TGVKDeshadeDownsampledWidget::TGVKDeshadeDownsampledWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::TGVKDeshadeDownsampledWidget)
{
    ui->setupUi(this);
    this->updateAlpha();

    this->shading_output_view = new ImageViewWidget("Denoised", this->ui->shading_frame);
    this->ui->shading_frame->layout()->addWidget(this->shading_output_view);

    this->denoised_output_view = new ImageViewWidget("Denoised", this->ui->denoised_frame);
    this->ui->denoised_frame->layout()->addWidget(this->denoised_output_view);

    this->mask_view = new ImageViewWidget("Mask", this->ui->mask_frame);
    this->ui->mask_frame->layout()->addWidget(this->mask_view);

    this->div_v_view = new ImageViewWidget("div v", this->ui->div_v_frame);
    this->ui->div_v_frame->layout()->addWidget(this->div_v_view);
}

TGVKDeshadeDownsampledWidget::~TGVKDeshadeDownsampledWidget()
{
    delete ui;
}


void TGVKDeshadeDownsampledWidget::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(this, &BaseModuleWidget::fireWorkerFinished,
            this, [this]() {
        this->ui->stop_button->setEnabled(false);
    });

    this->shading_output_view->registerCrosshairSubmodule(image_widget);
    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this->shading_output_view, &ImageViewWidget::sliceIndexChanged);
    this->denoised_output_view->registerCrosshairSubmodule(image_widget);
    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this->denoised_output_view, &ImageViewWidget::sliceIndexChanged);
    this->mask_view->registerCrosshairSubmodule(image_widget);
    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this->mask_view, &ImageViewWidget::sliceIndexChanged);
    this->div_v_view->registerCrosshairSubmodule(image_widget);
    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this->div_v_view, &ImageViewWidget::sliceIndexChanged);
}

void TGVKDeshadeDownsampledWidget::setIterationFinishedCallback(TGVKDeshadeDownsampledProcessor::IterationFinished iteration_finished_callback)
{
    this->iteration_finished_callback = [this, iteration_finished_callback](uint iteration_index, uint iteration_count,
            ITKImage u, ITKImage l, ITKImage r){
        iteration_finished_callback(iteration_index, iteration_count, r);

        this->shading_output_view->fireImageChange(l);
        this->denoised_output_view->fireImageChange(u);

        return this->stop_after_next_iteration;
    };
}

ITKImage TGVKDeshadeDownsampledWidget::processImage(ITKImage image)
{
    const uint order = this->ui->order_spinbox->value();
    ITKImage::PixelType* alpha = new ITKImage::PixelType[order];
    for(int k = 0; k < order; k++)
        alpha[k] = this->alpha_spinboxes.at(k)->value() * this->ui->alpha_factor_spinbox->value();

    const float lambda = this->ui->lambda_spinbox->value();
    const uint iteration_count = this->ui->iteration_count_spinbox->value();
    const uint paint_iteration_interval = this->ui->paint_iteration_interval_spinbox->value();

    const bool set_negative_values_to_zero = this->ui->set_negative_values_to_zero_checkbox->isChecked();
    auto mask = this->mask_view->getImage();

    if(!mask.isNull() &&
            (!mask.hasSameSize(image)))
    {
        throw std::runtime_error("the given mask has different size than the input image");
    }

    const bool add_background_back = this->ui->add_background_back_checkbox->isChecked();

    const ITKImage::PixelType downsampling_factor = this->ui->downsampling_factor_spinbox->value();

    const bool calculate_div_v = this->ui->calculate_div_v_checkbox->isChecked();

    ITKImage denoised_image = ITKImage();
    ITKImage shading_image = ITKImage();
    ITKImage deshaded_image = ITKImage();
    ITKImage div_v_image = ITKImage();
    TGVKDeshadeDownsampledProcessor::processTGVKL1Cuda(
              image,
              downsampling_factor,
              lambda,

              order,
              alpha,

              iteration_count,
              mask,
              set_negative_values_to_zero,
              add_background_back,

              paint_iteration_interval,
              this->iteration_finished_callback,

              denoised_image,
              shading_image,
              deshaded_image,
              div_v_image,
              calculate_div_v);
    delete[] alpha;
    this->denoised_output_view->setImage(denoised_image);
    this->shading_output_view->setImage(shading_image);
    this->div_v_view->setImage(div_v_image);
    return deshaded_image;
}


void TGVKDeshadeDownsampledWidget::on_perform_button_clicked()
{
    this->stop_after_next_iteration = false;
    this->processInWorkerThread();
    this->ui->stop_button->setEnabled(true);
}

void TGVKDeshadeDownsampledWidget::on_stop_button_clicked()
{
    this->stop_after_next_iteration = true;
}

void TGVKDeshadeDownsampledWidget::on_save_second_output_button_clicked()
{
    auto image = this->shading_output_view->getImage();
    if(image.isNull())
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save volume file");
    if(file_name.isNull())
        return;

    image.write(file_name.toStdString());
}

void TGVKDeshadeDownsampledWidget::on_load_mask_button_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "open volume file");
    if(file_name == QString::null || !QFile(file_name).exists())
        return;

    this->mask_view->setImage(ITKImage::read(file_name.toStdString()));
}

void TGVKDeshadeDownsampledWidget::on_save_denoised_button_clicked()
{
    auto image = this->denoised_output_view->getImage();
    if(image.isNull())
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save volume file");
    if(file_name.isNull())
        return;

    image.write(file_name.toStdString());
}

void TGVKDeshadeDownsampledWidget::on_clear_mask_button_clicked()
{
    this->mask_view->setImage(ITKImage());
}

void TGVKDeshadeDownsampledWidget::on_order_spinbox_editingFinished()
{
    this->updateAlpha();
}

void TGVKDeshadeDownsampledWidget::updateAlpha()
{
    delete this->ui->alpha_groupbox;
    this->ui->alpha_groupbox = new QGroupBox("Alpha", this->ui->alpha_groupbox_frame);
    this->ui->alpha_groupbox->setLayout(new QVBoxLayout());
    this->ui->alpha_groupbox_frame->layout()->addWidget(this->ui->alpha_groupbox);

    this->alpha_spinboxes.clear();

    const int order = this->ui->order_spinbox->value();

    TGVKDeshadeProcessor::updateAlpha(order, [this](uint alpha_element){
        this->addAlpha(alpha_element);
    });
}

void TGVKDeshadeDownsampledWidget::addAlpha(uint alpha_element)
{
    auto alpha_groupbox = new QGroupBox(this->ui->alpha_groupbox);
    this->ui->alpha_groupbox->layout()->addWidget(alpha_groupbox);
    alpha_groupbox->setLayout(new QHBoxLayout());

    auto spinbox = new QDoubleSpinBox(alpha_groupbox);
    this->alpha_spinboxes.push_back(spinbox);

    spinbox->setMinimum(1e-8);
    spinbox->setMaximum(1e5);
    spinbox->setDecimals(12);
    spinbox->setValue(alpha_element);
    spinbox->setSingleStep(0.01);
    alpha_groupbox->layout()->addWidget(spinbox);
    alpha_groupbox->setTitle("Alpha" + QString::number(this->alpha_spinboxes.size() - 1));
}

void TGVKDeshadeDownsampledWidget::on_save_div_v_button_clicked()
{
    auto image = this->div_v_view->getImage();
    if(image.isNull())
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save volume file");
    if(file_name.isNull())
        return;

    image.write(file_name.toStdString());
}
