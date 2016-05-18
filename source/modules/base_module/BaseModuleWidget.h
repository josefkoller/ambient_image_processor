#ifndef BASEMODULEWIDGET_H
#define BASEMODULEWIDGET_H

#include <functional>
#include <thread>

#include "ITKImage.h"
#include "ImageWidget.h"

#include <QWidget>

class BaseModuleWidget : public QWidget
{
    Q_OBJECT
public:
    BaseModuleWidget(QWidget *parent);

    typedef std::function<void(ITKImage)> ResultProcessor;
    typedef std::function<ITKImage()> SourceImageFetcher;

    void setSourceImageFetcher(SourceImageFetcher source_image_fetcher);
    void setResultProcessor(ResultProcessor result_processor);
private:
    SourceImageFetcher source_image_fetcher;
    ResultProcessor result_processor;

    std::thread* worker_thread;
signals:
    void fireWorkerFinished();
private slots:
    void handleWorkerFinished();

protected:
    virtual ITKImage processImage(ITKImage image);
    void processInWorkerThread();

public:
    virtual void registerModule(ImageWidget* image_widget);
};

#endif // BASEMODULEWIDGET_H
