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

#ifndef TGVDESHADEMASKEDWIDGET_H
#define TGVDESHADEMASKEDWIDGET_H

#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"
#include "MaskWidget.h"

namespace Ui {
class TGVDeshadeMaskedWidget;
}

#include "TGVDeshadeMaskedProcessor.h"

class TGVDeshadeMaskedWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    TGVDeshadeMaskedWidget(QString title, QWidget *parent);
    ~TGVDeshadeMaskedWidget();

protected:
    Ui::TGVDeshadeMaskedWidget *ui;

    ImageViewWidget* shading_output_view;
    ImageViewWidget* denoised_output_view;
    MaskWidget::MaskFetcher mask_fetcher;

    TGVDeshadeMaskedProcessor::IterationFinishedThreeImages iteration_finished_callback;
    bool stop_after_next_iteration;
public:
    void setIterationFinishedCallback(TGVDeshadeMaskedProcessor::IterationFinished iteration_finished_callback);

    void registerModule(ImageWidget *image_widget);
protected:
    virtual ITKImage processImage(ITKImage image);
private slots:
    void on_perform_button_clicked();
    void on_stop_button_clicked();
    void on_save_second_output_button_clicked();
    void on_save_denoised_button_clicked();
};

#endif // TGVDESHADEMASKEDWIDGET_H
