#ifndef BASEMODULEWIDGET_H
#define BASEMODULEWIDGET_H

#include <functional>
#include <thread>

#include "../../itk_image/ITKImage.h"

#include <QWidget>

class BaseModuleWidget : public QWidget
{
    Q_OBJECT
public:
    BaseModuleWidget();

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
};

#endif // BASEMODULEWIDGET_H
