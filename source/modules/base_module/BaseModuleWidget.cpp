#include "BaseModuleWidget.h"

BaseModuleWidget::BaseModuleWidget(ImageWidget *parent) :
    QWidget(parent),
    source_image_fetcher(nullptr),
    result_processor(nullptr),
    worker_thread(nullptr)
{
    connect(this, SIGNAL(fireWorkerFinished()),this, SLOT(handleWorkerFinished()));
}


void BaseModuleWidget::setSourceImageFetcher(SourceImageFetcher source_image_fetcher)
{
    this->source_image_fetcher = source_image_fetcher;
}

void BaseModuleWidget::setResultProcessor(ResultProcessor result_processor)
{
    this->result_processor = result_processor;
}

void BaseModuleWidget::processInWorkerThread()
{
    if(this->source_image_fetcher == nullptr) {
        std::cout << "source_image_fetcher not set" << std::endl;
        return;
    }
    if(this->result_processor == nullptr) {
        std::cout << "result_processor not set" << std::endl;
        return;
    }

    ITKImage source_image = this->source_image_fetcher();
    if(source_image.isNull()) {
        std::cout << "source_image_fetcher returned null source_image" << std::endl;
        return;
    }

    if(worker_thread != nullptr) {
        std::cout << "worker thread not finished" << std::endl;
        return;

    }

    this->worker_thread = new std::thread([=]() {
        ITKImage result_image = this->processImage(source_image);
        this->result_processor(result_image);
        emit this->fireWorkerFinished();
    });
}

ITKImage BaseModuleWidget::processImage(ITKImage image)
{
    return image;
}


void BaseModuleWidget::handleWorkerFinished()
{
    if(this->worker_thread != nullptr)
    {
        this->worker_thread->join();
        delete this->worker_thread;
        this->worker_thread = nullptr;
    }
}

void BaseModuleWidget::registerModule(ImageWidget* image_widget)
{
    this->setSourceImageFetcher([image_widget](){
        return ITKImage(image_widget->getImage());
    });
    this->setResultProcessor( [image_widget] (ITKImage image) {
        emit image_widget->getOutputWidget()->fireImageChange(image.getPointer());
    });
}
