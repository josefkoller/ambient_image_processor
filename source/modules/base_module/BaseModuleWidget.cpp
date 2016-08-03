#include "BaseModuleWidget.h"

BaseModuleWidget::BaseModuleWidget(QString title, QWidget *parent) :
    QWidget(parent),
    BaseModule(title),
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
        int duration = this->start_timestamp.msecsTo(QTime::currentTime());
        QString text = this->getTitle() + " already started " +
                QString::number(duration) + "ms ago";
        this->setStatusText(text);
        std::cout << text.toStdString() << std::endl;
        return;

    }

    this->setStatusText(this->getTitle() + " started");
    this->start_timestamp = QTime::currentTime();

    this->worker_thread = new std::thread([=]() {
        try
        {
            ITKImage result_image = this->processImage(source_image);

            if(this->calculatesResultImage())
                this->result_processor(result_image);

            int duration = this->start_timestamp.msecsTo(QTime::currentTime());
            QString text = this->getTitle() + " finished after " + QString::number(duration) + "ms";
            this->setStatusText(text);
            std::cout << text.toStdString() << std::endl;
        }
        catch(std::runtime_error exception)
        {
            std::cerr << "error in " << this->getTitle().toStdString() << ": " <<
                         exception.what() << std::endl;
            this->setStatusText("Error in " + this->getTitle() + ": " +  QString(exception.what()));
        }
        catch(itk::ExceptionObject exception)
        {
            std::cerr << "error in " << this->getTitle().toStdString() << ": " <<
                         exception.what() << std::endl;
            this->setStatusText("Error in " + this->getTitle() + ": " +  QString(exception.what()));
        }
        catch(std::exception exception)
        {
            std::cerr << "error in " << this->getTitle().toStdString() << ": " <<
                         exception.what() << std::endl;
            this->setStatusText("Error in " + this->getTitle() + ". see console output");
        }
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
    BaseModule::registerModule(image_widget);

    this->source_image_fetcher = [image_widget](){
        return image_widget->getImage();
    };
    this->result_processor = [image_widget] (ITKImage image) {
        emit image_widget->getOutputWidget()->fireImageChange(image);
    };
}

ITKImage BaseModuleWidget::getSourceImage() const
{
    if(this->source_image_fetcher == nullptr)
        return ITKImage();
    return this->source_image_fetcher();
}

bool BaseModuleWidget::calculatesResultImage() const
{
    return true;
}

void BaseModuleWidget::setSourceImageFetcher(SourceImageFetcher source_image_fetcher)
{
    this->source_image_fetcher = source_image_fetcher;
}

void BaseModuleWidget::setResultProcessor(ResultProcessor result_processor)
{
    this->result_processor = result_processor;
}
