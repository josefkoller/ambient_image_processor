#include "TGVWidget.h"
#include "ui_TGVWidget.h"

TGVWidget::TGVWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::TGVWidget),
    source_image_fetcher(nullptr),
    result_processor(nullptr),
    iteration_finished_callback(nullptr),
    worker_thread(nullptr)
{
    ui->setupUi(this);

    connect(this, SIGNAL(fireWorkerFinished()),
            this, SLOT(handleWorkerFinished()));
}

TGVWidget::~TGVWidget()
{
    delete ui;
}

void TGVWidget::perform(Processor processor)
{
    if(this->source_image_fetcher == nullptr ||
            this->result_processor == nullptr)
        return;

    const float alpha0 = this->ui->alpha0_spinbox->value();
    const float alpha1 = this->ui->alpha1_spinbox->value();
    const float lambda = this->ui->lambda_spinbox->value();
    const uint iteration_count = this->ui->iteration_count_spinbox->value();

    TGVProcessor::itkImage::Pointer source_image = this->source_image_fetcher();
    if(source_image.IsNull())
        return;


    if(worker_thread != nullptr)
        return;

    this->worker_thread = new std::thread([=]() {

        TGVProcessor::itkImage::Pointer result_image = processor(
                    source_image, lambda, alpha0, alpha1, iteration_count);
        this->result_processor(result_image);

        emit this->fireWorkerFinished();
    });

}

void TGVWidget::setSourceImageFetcher(SourceImageFetcher source_image_fetcher)
{
    this->source_image_fetcher = source_image_fetcher;
}

void TGVWidget::setResultProcessor(ResultProcessor result_processor)
{
    this->result_processor = result_processor;
}

void TGVWidget::setIterationFinishedCallback(TGVProcessor::IterationFinished iteration_finished_callback)
{
    this->iteration_finished_callback = iteration_finished_callback;
}

void TGVWidget::on_perform_button_cpu_clicked()
{
    this->perform([this](TGVProcessor::itkImage::Pointer source, float lambda, float alpha0, float alpha1,
                  uint iteration_count) {
        return TGVProcessor::processTVL2CPU(source,lambda, iteration_count, this->iteration_finished_callback);
    });
}


void TGVWidget::on_perform_button_clicked()
{
    this->perform([this](TGVProcessor::itkImage::Pointer source, float lambda, float alpha0, float alpha1,
                  uint iteration_count) {
        return TGVProcessor::processTVL2GPU(source,lambda, iteration_count, this->iteration_finished_callback);
    });
}


void TGVWidget::handleWorkerFinished()
{
    if(this->worker_thread != nullptr)
    {
        this->worker_thread->join();
        delete this->worker_thread;
        this->worker_thread = nullptr;
    }
}
