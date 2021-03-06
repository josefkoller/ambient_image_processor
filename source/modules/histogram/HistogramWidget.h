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

#ifndef HISTOGRAMWIDGET_H
#define HISTOGRAMWIDGET_H

#include <QWidget>
#include "BaseModuleWidget.h"
#include "MaskWidget.h"

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
    MaskWidget::MaskFetcher mask_fetcher;

    static const uint line_width;
    static const QColor line_color;
private slots:
    void histogram_mouse_move(QMouseEvent* position);
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
    void estimateBandwidthAndWindow(const ITKImage& image, const ITKImage& mask,
                                    ITKImage::PixelType& window_from,
                                    ITKImage::PixelType& window_to,
                                    ITKImage::PixelType& kernel_bandwidth);

    void calculateHistogram();
public:
    virtual void registerModule(ImageWidget* image_widget);

    void write(QString filename);
    ITKImage::PixelType getEntropy();
    void calculateHistogramSync();
signals:
    void fireImageRepaint();
    void fireHistogramChanged(std::vector<double> intensities,
                              std::vector<double> probabilities);
    void fireEntropyLabelTextChange(QString text);
    void fireKernelBandwidthAndWindowChange(double kernel_bandwidth,
                                            double window_from,
                                            double window_to);
protected:
    bool calculatesResultImage() const;

private slots:

    void handleHistogramChanged(std::vector<double> intensities,
                              std::vector<double> probabilities);
    void handleEntropyLabelTextChange(QString text);
    void on_copy_to_clipboard_button_clicked();
    void on_save_button_clicked();
    void on_use_mask_module_checkbox_clicked();
    void on_estimate_bandwidth_and_window_checkbox_toggled(bool checked);

    void handleKernelBandwidthAndWindowChange(double kernel_bandwidth,
                                            double window_from,
                                            double window_to);
};

#endif // HISTOGRAMWIDGET_H
