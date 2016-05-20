#include "MainWindow.h"
#include "ui_MainWindow.h"

#include <QDateTime>
#include <QPainter>

#include "ITKImage.h"

MainWindow::MainWindow(std::string image_path) :
    QMainWindow(NULL),
    ui(new Ui::MainWindow),
    source_image_path(image_path)
{
    ui->setupUi(this);

    if(QFile(QString::fromStdString(image_path)).exists())
    {
        this->ui->image_widget->setImage(ITKImage::read(image_path));
    }

   // this->ui->image_widget->showSliceControl();
    this->ui->image_widget->setOutputWidget(this->ui->output_widget);
    this->ui->output_widget->connectSliceControlTo(this->ui->image_widget);
    this->ui->output_widget->connectModule("Line Profile", this->ui->image_widget);
    this->ui->output_widget->setPage(0);
}

MainWindow::~MainWindow()
{
    delete ui;
}
