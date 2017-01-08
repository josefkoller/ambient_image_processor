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

#ifndef TGVKDeshadeDownsampledWIDGET_H
#define TGVKDeshadeDownsampledWIDGET_H

#include "BaseModuleWidget.h"
#include "TGVKDeshadeDownsampledProcessor.h"
#include "ImageViewWidget.h"
#include "MaskWidget.h"

#include <QVector>
#include <QDoubleSpinBox>

namespace Ui {
class TGVKDeshadeDownsampledWidget;
}

class TGVKDeshadeDownsampledWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    TGVKDeshadeDownsampledWidget(QString title, QWidget *parent);
    ~TGVKDeshadeDownsampledWidget();

private slots:
    void on_perform_button_clicked();
    void on_stop_button_clicked();
    void on_save_second_output_button_clicked();
    void on_save_denoised_button_clicked();


    void on_order_spinbox_editingFinished();

    void on_save_div_v_button_clicked();

private:
    Ui::TGVKDeshadeDownsampledWidget *ui;

protected:
    ImageViewWidget* shading_output_view;
    ImageViewWidget* denoised_output_view;
    ImageViewWidget* div_v_view;
    MaskWidget::MaskFetcher mask_fetcher;

    TGVKDeshadeDownsampledProcessor::IterationFinishedThreeImages iteration_finished_callback;
    bool stop_after_next_iteration;
    QVector<QDoubleSpinBox*> alpha_spinboxes;

    virtual ITKImage processImage(ITKImage image);

    void addAlpha(uint index);
    void updateAlpha();
public:
    void setIterationFinishedCallback(TGVKDeshadeDownsampledProcessor::IterationFinished iteration_finished_callback);

    void registerModule(ImageWidget *image_widget);
};

#endif // TGVKDeshadeDownsampledWIDGET_H
