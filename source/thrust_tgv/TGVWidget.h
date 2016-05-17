#ifndef TGVWIDGET_H
#define TGVWIDGET_H

#include <QWidget>

namespace Ui {
class TGVWidget;
}

#include "TGVProcessor.h"

class TGVWidget : public QWidget
{
    Q_OBJECT

public:
    explicit TGVWidget(QWidget *parent = 0);
    ~TGVWidget();

    typedef std::function<void(TGVProcessor::itkImage::Pointer)> ResultProcessor;
    typedef std::function<TGVProcessor::itkImage::Pointer()> SourceImageFetcher;
private slots:
    void on_perform_button_clicked();

    void on_perform_button_cpu_clicked();

private:
    Ui::TGVWidget *ui;

    SourceImageFetcher source_image_fetcher;
    ResultProcessor result_processor;

    typedef std::function<TGVProcessor::itkImage::Pointer(
            TGVProcessor::itkImage::Pointer source, float lambda, float alpha0, float alpha1,
            uint iteration_count)> Processor;
    void perform(Processor processor);
public:
    void setSourceImageFetcher(SourceImageFetcher source_image_fetcher);
    void setResultProcessor(ResultProcessor result_processor);
};

#endif // TGVWIDGET_H
