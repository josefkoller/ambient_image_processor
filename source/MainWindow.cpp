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

#include "MainWindow.h"
#include "ui_MainWindow.h"

#include <QDateTime>
#include <QPainter>

#include "ITKImage.h"

MainWindow::MainWindow(std::string image_path, std::string image_path2) :
    QMainWindow(NULL),
    ui(new Ui::MainWindow),
    source_image_path(image_path)
{
    ui->setupUi(this);

    if(QFile(QString::fromStdString(image_path)).exists())
    {
        this->ui->image_widget->setImage(ITKImage::read(image_path));
    }

    if(QFile(QString::fromStdString(image_path2)).exists())
    {
        this->ui->output_widget->setImage(ITKImage::read(image_path2));
    }

    this->ui->image_widget->setOutputWidget(this->ui->output_widget);
    this->ui->output_widget->connectModule("Slice Control", this->ui->image_widget);
    this->ui->output_widget->connectModule("Line Profile", this->ui->image_widget);
    this->ui->output_widget->setPage(0);

    /*
    this->ui->image_widget->setOutputWidget2(this->ui->output_widget2);
    this->ui->output_widget2->connectModule("Slice Control", this->ui->image_widget);
    this->ui->output_widget2->connectModule("Line Profile", this->ui->image_widget);
    this->ui->output_widget2->setPage(0);
    */

    this->showMaximized();
}

MainWindow::~MainWindow()
{
    delete ui;
}
