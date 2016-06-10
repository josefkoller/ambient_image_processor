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
