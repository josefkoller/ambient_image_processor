/*
    Ambient Image Processor - A tool to perform several imaging tasks
    
    Copyright (C) 2016 Josef Koller

    https://github.com/josefkoller/ambient_image_processor    
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

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
            this->setStatusText(this->getTitle() + " finished after " + QString::number(duration) + "ms");
        }
        catch(itk::ExceptionObject exception)
        {
            std::ostringstream stream;
            exception.Print(stream);
            this->setStatusText("error in " + this->getTitle() + ": " +  QString::fromStdString(stream.str()));
        }
        catch(std::runtime_error exception)
        {
            this->setStatusText("Error in " + this->getTitle() + ": " +  QString(exception.what()));
        }
        catch(std::exception exception)
        {
            this->setStatusText("Error in " + this->getTitle() + ": " +  QString(exception.what()));
        }
        catch(char const* text)
        {
            this->setStatusText("Error in " + this->getTitle() + ": " + QString(text));
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
