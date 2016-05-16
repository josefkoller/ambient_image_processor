#ifndef NONLOCALGRADIENTWIDGET_H
#define NONLOCALGRADIENTWIDGET_H

#include <QWidget>

#include <functional>

#include "NonLocalGradientProcessor.h"

namespace Ui {
class NonLocalGradientWidget;
}

class NonLocalGradientWidget : public QWidget
{
    Q_OBJECT

public:
    explicit NonLocalGradientWidget(QWidget *parent = 0);
    ~NonLocalGradientWidget();

    typedef std::function<void(NonLocalGradientProcessor::Image::Pointer)> ResultProcessor;
    typedef std::function<NonLocalGradientProcessor::Image::Pointer()> SourceImageFetcher;
private slots:
    void on_kernel_size_spinbox_valueChanged(int arg1);

    void on_sigma_spinbox_valueChanged(double arg1);

    void on_perform_button_clicked();

private:
    Ui::NonLocalGradientWidget *ui;
    ResultProcessor result_processor;
    NonLocalGradientProcessor::Image::Pointer source_image;
    SourceImageFetcher source_image_fetcher;
public:
    void setResultProcessor(ResultProcessor result_processor);
    void setSourceImageFetcher(SourceImageFetcher source_image_fetcher);

    float getKernelSigma() const;
    uint getKernelSize() const;


};

#endif // NONLOCALGRADIENTWIDGET_H
