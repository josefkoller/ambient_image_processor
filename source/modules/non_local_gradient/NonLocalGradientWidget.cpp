#include "NonLocalGradientWidget.h"
#include "ui_NonLocalGradientWidget.h"

NonLocalGradientWidget::NonLocalGradientWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::NonLocalGradientWidget),
    result_processor(nullptr),
    source_image_fetcher(nullptr)
{
    ui->setupUi(this);
}

NonLocalGradientWidget::~NonLocalGradientWidget()
{
    delete ui;
}

void NonLocalGradientWidget::on_kernel_size_spinbox_valueChanged(int arg1)
{
    auto kernel_size = this->ui->kernel_size_spinbox->value();
    this->ui->kernel_view->setKernelSize(kernel_size);
}

void NonLocalGradientWidget::on_sigma_spinbox_valueChanged(double arg1)
{
    auto sigma = this->ui->sigma_spinbox->value();
    this->ui->kernel_view->setSigma(sigma);
}

void NonLocalGradientWidget::on_perform_button_clicked()
{
    if(this->result_processor == nullptr || this->source_image_fetcher == nullptr)
        return;

    uint kernel_size = this->ui->kernel_size_spinbox->value();
    float kernel_sigma = this->ui->sigma_spinbox->value();

    NonLocalGradientProcessor::Image::Pointer result_image = NonLocalGradientProcessor::process(
                this->source_image_fetcher(), kernel_size, kernel_sigma);
    this->result_processor(result_image);
}

void NonLocalGradientWidget::setResultProcessor(ResultProcessor result_processor)
{
    this->result_processor = result_processor;
}

void NonLocalGradientWidget::setSourceImageFetcher(SourceImageFetcher source_image_fetcher)
{
    this->source_image_fetcher = source_image_fetcher;
}

float NonLocalGradientWidget::getKernelSigma() const
{
    return this->ui->sigma_spinbox->value();
}
uint NonLocalGradientWidget::getKernelSize() const
{
    return this->ui->kernel_size_spinbox->value();
}
