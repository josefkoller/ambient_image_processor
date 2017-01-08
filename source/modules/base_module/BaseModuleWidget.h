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

#ifndef BASEMODULEWIDGET_H
#define BASEMODULEWIDGET_H

#include <functional>
#include <thread>

#include "ITKImage.h"
#include "ImageWidget.h"

#include <QWidget>
#include <QTime>

#include "BaseModule.h"

class BaseModuleWidget : public QWidget, public BaseModule
{
    Q_OBJECT
public:
    BaseModuleWidget(QString title, QWidget *parent);

    typedef std::function<void(ITKImage)> ResultProcessor;
    typedef std::function<ITKImage()> SourceImageFetcher;
private:
    SourceImageFetcher source_image_fetcher;
    ResultProcessor result_processor;

    QTime start_timestamp;
    std::thread* worker_thread;
signals:
    void fireWorkerFinished();
private slots:
    void handleWorkerFinished();

protected:
    virtual ITKImage processImage(ITKImage image);
    void processInWorkerThread();

    ITKImage getSourceImage() const;
    virtual bool calculatesResultImage() const;

public:
    virtual void registerModule(ImageWidget* image_widget);

    void setSourceImageFetcher(SourceImageFetcher source_image_fetcher);
    void setResultProcessor(ResultProcessor result_processor);

};

#endif // BASEMODULEWIDGET_H
