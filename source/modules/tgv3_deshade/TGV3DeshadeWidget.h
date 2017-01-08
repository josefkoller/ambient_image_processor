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

#ifndef TGV3DESHADEWIDGET_H
#define TGV3DESHADEWIDGET_H

#include "BaseModuleWidget.h"
#include "TGV3DeshadeProcessor.h"
#include "ImageViewWidget.h"

namespace Ui {
class TGV3DeshadeWidget;
}

class TGV3DeshadeWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    TGV3DeshadeWidget(QString title, QWidget *parent);
    ~TGV3DeshadeWidget();

private slots:
    void on_perform_button_clicked();
    void on_stop_button_clicked();
    void on_save_second_output_button_clicked();
    void on_load_mask_button_clicked();
    void on_save_denoised_button_clicked();

    void on_clear_mask_button_clicked();

private:
    Ui::TGV3DeshadeWidget *ui;

protected:
    ImageViewWidget* shading_output_view;
    ImageViewWidget* denoised_output_view;
    ImageViewWidget* mask_view;

    TGV3DeshadeProcessor::IterationFinishedThreeImages iteration_finished_callback;
    bool stop_after_next_iteration;

    virtual ITKImage processImage(ITKImage image);
public:
    void setIterationFinishedCallback(TGV3DeshadeProcessor::IterationFinished iteration_finished_callback);

    void registerModule(ImageWidget *image_widget);
};

#endif // TGV3DESHADEWIDGET_H
