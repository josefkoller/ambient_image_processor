#ifndef HISTOGRAMWIDGET_H
#define HISTOGRAMWIDGET_H

#include <QWidget>
#include "BaseModuleWidget.h"

namespace Ui {
class HistogramWidget;
}

class HistogramWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit HistogramWidget(QWidget *parent = 0);
    ~HistogramWidget();

private:
    Ui::HistogramWidget *ui;

    typedef ITKImage::InnerITKImage Image;
    Image::Pointer image;
private slots:
    void histogram_mouse_move(QMouseEvent* event);
    void handleImageChanged(Image::Pointer image);

    void on_histogram_bin_count_spinbox_valueChanged(int arg1);

    void on_window_from_spinbox_valueChanged(double arg1);

    void on_window_to_spinbox_valueChanged(double arg1);

    void on_fromMinimumButton_clicked();

    void on_toMaximumButton_clicked();

private:
    void calculateHistogram();
public:
    virtual void registerModule(ImageWidget* image_widget);

signals:
    void fireImageRepaint();
};

#endif // HISTOGRAMWIDGET_H