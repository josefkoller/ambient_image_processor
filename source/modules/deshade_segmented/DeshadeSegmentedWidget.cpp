#include "DeshadeSegmentedWidget.h"
#include "ui_DeshadeSegmentedWidget.h"

#include "DeshadeSegmentedProcessor.h"

DeshadeSegmentedWidget::DeshadeSegmentedWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::DeshadeSegmentedWidget),
    segment_fetcher(nullptr),
    label_image_fetcher(nullptr)
{
    ui->setupUi(this);
}

DeshadeSegmentedWidget::~DeshadeSegmentedWidget()
{
    delete ui;
}

void DeshadeSegmentedWidget::on_perform_button_clicked()
{
    this->processInWorkerThread();
}

ITKImage DeshadeSegmentedWidget::processImage(ITKImage image)
{
    if(this->segment_fetcher == nullptr ||
       this->label_image_fetcher == nullptr)
        return ITKImage();

    Segments segments = this->segment_fetcher();
    LabelImage::Pointer label_image = this->label_image_fetcher();
    Image::Pointer source_image = image.getPointer();

    if(source_image.IsNull() || label_image.IsNull() || segments.size() == 0)
        return ITKImage();

    float lambda = this->ui->lambda_spinbox->value();

    Image::Pointer reflectance_image;
    Image::Pointer shading_image = DeshadeSegmentedProcessor::process(
                source_image, lambda, segments, label_image, reflectance_image);

    return ITKImage(shading_image);
}

void DeshadeSegmentedWidget::setSegmentsFetcher(SegmentsFetcher segment_fetcher)
{
    this->segment_fetcher = segment_fetcher;
}

void DeshadeSegmentedWidget::setLabelImageFetcher(LabelImageFetcher label_image_fetcher)
{
    this->label_image_fetcher = label_image_fetcher;
}
