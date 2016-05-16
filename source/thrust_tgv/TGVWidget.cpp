#include "TGVWidget.h"
#include "ui_TGVWidget.h"

TGVWidget::TGVWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::TGVWidget),
    source_image_fetcher(nullptr),
    result_processor(nullptr)
{
    ui->setupUi(this);
}

TGVWidget::~TGVWidget()
{
    delete ui;
}

void TGVWidget::on_perform_button_clicked()
{
    if(this->source_image_fetcher == nullptr ||
            this->result_processor == nullptr)
        return;

    const float alpha0 = this->ui->alpha0_spinbox->value();
    const float alpha1 = this->ui->alpha1_spinbox->value();
    const float lambda = this->ui->alpha1_spinbox->value();
    const uint iteration_count = this->ui->iteration_count_spinbox->value();

    TGVProcessor::itkImage::Pointer source_image = this->source_image_fetcher();
    if(source_image.IsNull())
        return;

    TGVProcessor::itkImage::Pointer result_image =
            TGVProcessor::processTVL2(source_image, lambda, iteration_count);

    this->result_processor(result_image);
}

void TGVWidget::setSourceImageFetcher(SourceImageFetcher source_image_fetcher)
{
    this->source_image_fetcher = source_image_fetcher;
}

void TGVWidget::setResultProcessor(ResultProcessor result_processor)
{
    this->result_processor = result_processor;
}
