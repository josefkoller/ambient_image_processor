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

#ifndef TGVLAMBDASWIDGET_H
#define TGVLAMBDASWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"
#include "TGVLambdasProcessor.h"

namespace Ui {
class TGVLambdasWidget;
}

class TGVLambdasWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit TGVLambdasWidget(QString title, QWidget *parent = 0);
    ~TGVLambdasWidget();

    virtual void registerModule(ImageWidget* image_widget);

private slots:
    void on_load_button_clicked();

    void on_perform_button_clicked();

    void on_stop_button_clicked();

private:
    Ui::TGVLambdasWidget *ui;

    ITKImage lambdas_image;
    ImageViewWidget* lambdas_widget;

    bool stop_after_next_iteration;

    TGVLambdasProcessor::IterationFinished iteration_finished_callback;
public:
    void setIterationFinishedCallback(TGVLambdasProcessor::IterationFinished iteration_finished_callback);

protected:
    virtual ITKImage processImage(ITKImage image);
};

#endif // TGVLAMBDASWIDGET_H
