#ifndef HISTOGRAMWIDGET_H
#define HISTOGRAMWIDGET_H

#include <QWidget>
#include "BaseModuleWidget.h"

Q_DECLARE_METATYPE(std::vector<double>)

namespace Ui {
class HistogramWidget;
}

class HistogramWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit HistogramWidget(QString title, QWidget *parent = 0);
    ~HistogramWidget();

private:
    Ui::HistogramWidget *ui;

    ITKImage image;
private slots:
    void histogram_mouse_move(QMouseEvent* event);
    void handleImageChanged(ITKImage image);

    void on_window_from_spinbox_valueChanged(double arg1);

    void on_window_to_spinbox_valueChanged(double arg1);

    void on_fromMinimumButton_clicked();

    void on_toMaximumButton_clicked();

    void on_kernel_bandwidth_valueChanged(double arg1);

    void on_uniform_kernel_checkbox_toggled(bool checked);

    void on_spectrum_bandwidth_spinbox_valueChanged(int arg1);

    void on_epanechnik_kernel_checkbox_toggled(bool checked);

    void on_cosine_kernel_checkbox_toggled(bool checked);

    void on_gaussian_kernel_checkbox_toggled(bool checked);

private:
    ITKImage processImage(ITKImage image);
    void calculateEntropy(const std::vector<double>& probabilities);
    void calculateHistogram();
public:
    virtual void registerModule(ImageWidget* image_widget);

signals:
    void fireImageRepaint();
    void fireHistogramChanged(std::vector<double> intensities,
                              std::vector<double> probabilities);
    void fireEntropyLabelTextChange(QString text);
protected:
    bool calculatesResultImage() const;

private slots:

    void handleHistogramChanged(std::vector<double> intensities,
                              std::vector<double> probabilities);
    void handleEntropyLabelTextChange(QString text);
    void on_copy_to_clipboard_button_clicked();
};

#endif // HISTOGRAMWIDGET_H
