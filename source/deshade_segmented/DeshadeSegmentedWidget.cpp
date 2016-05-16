#include "DeshadeSegmentedWidget.h"
#include "ui_DeshadeSegmentedWidget.h"

#include "DeshadeSegmentedProcessor.h"

DeshadeSegmentedWidget::DeshadeSegmentedWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::DeshadeSegmentedWidget),
    segment_fetcher(nullptr),
    label_image_fetcher(nullptr),
    source_image_fetcher(nullptr),
    result_processor(nullptr)
{
    ui->setupUi(this);
}

DeshadeSegmentedWidget::~DeshadeSegmentedWidget()
{
    delete ui;
}

void DeshadeSegmentedWidget::on_perform_button_clicked()
{
    if(this->segment_fetcher == nullptr ||
       this->label_image_fetcher == nullptr ||
       this->source_image_fetcher == nullptr ||
       this->result_processor == nullptr)
        return;

    float lambda = this->ui->lambda_spinbox->value();
    Segments segments = this->segment_fetcher();
    LabelImage::Pointer label_image = this->label_image_fetcher();
    Image::Pointer source_image = this->source_image_fetcher();

    if(source_image.IsNull() || label_image.IsNull())
        return;

    Image::Pointer reflectance_image;
    Image::Pointer shading_image = DeshadeSegmentedProcessor::process(
                source_image, lambda, segments, label_image, reflectance_image);

    this->result_processor(shading_image, reflectance_image);
}

void DeshadeSegmentedWidget::setSegmentsFetcher(SegmentsFetcher segment_fetcher)
{
    this->segment_fetcher = segment_fetcher;
}

void DeshadeSegmentedWidget::setLabelImageFetcher(LabelImageFetcher label_image_fetcher)
{
    this->label_image_fetcher = label_image_fetcher;
}

void DeshadeSegmentedWidget::setSourceImageFetcher(SourceImageFetcher source_image_fetcher)
{
    this->source_image_fetcher = source_image_fetcher;
}

void DeshadeSegmentedWidget::setResultProcessor(ResultProcessor result_processor)
{
    this->result_processor = result_processor;
}
