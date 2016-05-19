#include "BaseModuleWidget.h"

BaseModuleWidget::BaseModuleWidget(QWidget *parent) :
    QWidget(parent),
    source_image_fetcher(nullptr),
    result_processor(nullptr),
    worker_thread(nullptr)
{
    connect(this, SIGNAL(fireWorkerFinished()),this, SLOT(handleWorkerFinished()));
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
    this->source_image_fetcher = [image_widget](){
        return ITKImage(image_widget->getImage());
    };
    this->result_processor = [image_widget] (ITKImage image) {
        emit image_widget->getOutputWidget()->fireImageChange(image.getPointer());
    };
    this->status_text_processor = [image_widget] (QString text) {
        emit image_widget->fireStatusTextChange(text);
    };
}

ITKImage BaseModuleWidget::getSourceImage() const
{
    if(this->source_image_fetcher == nullptr)
        return ITKImage();
    return this->source_image_fetcher();
}

void BaseModuleWidget::setStatusText(QString text)
{
    if(this->status_text_processor != nullptr)
        this->status_text_processor(text);
}
