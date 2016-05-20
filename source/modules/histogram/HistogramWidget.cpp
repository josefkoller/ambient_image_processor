#include "HistogramWidget.h"
#include "ui_HistogramWidget.h"

#include "HistogramProcessor.h"

#include "ITKToQImageConverter.h"

HistogramWidget::HistogramWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::HistogramWidget)
{
    ui->setupUi(this);

    this->ui->custom_plot_widget->setMouseTracking(true);
    connect(this->ui->custom_plot_widget, &QCustomPlot::mouseMove,
            this, &HistogramWidget::histogram_mouse_move);
}

HistogramWidget::~HistogramWidget()
{
    delete ui;
}

void HistogramWidget::histogram_mouse_move(QMouseEvent* event)
{
    if(this->image.IsNull())
        return;

    QPoint position = event->pos();
    double pixel_value = this->ui->custom_plot_widget->xAxis->pixelToCoord(position.x());

    QString text = QString("pixel value at ") +
            QString::number(position.x()) +
            " | " +
            QString::number(position.y()) +
            " = " +
            QString::number(pixel_value);

    this->setStatusText(text);
}

void HistogramWidget::registerModule(ImageWidget* image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(image_widget, &ImageWidget::imageChanged,
            this, &HistogramWidget::handleImageChanged);
    connect(this, &HistogramWidget::fireImageRepaint,
            image_widget, &ImageWidget::handleRepaintImage);
}

void HistogramWidget::handleImageChanged(Image::Pointer image)
{
    this->image = image;
    this->calculateHistogram();
}

void HistogramWidget::calculateHistogram()
{
    if(this->image.IsNull())
        return;

    int bin_count = this->ui->histogram_bin_count_spinbox->value();

    Image::PixelType window_from = this->ui->window_from_spinbox->value();
    Image::PixelType window_to = this->ui->window_to_spinbox->value();

    std::vector<double> intensities;
    std::vector<double> probabilities;
    HistogramProcessor::calculate(this->image,
                                  bin_count,
                                  window_from, window_to,
                                  intensities, probabilities);

    this->ui->custom_plot_widget->xAxis->setLabel("intensity");
    this->ui->custom_plot_widget->xAxis->setNumberPrecision(8);
    this->ui->custom_plot_widget->xAxis->setOffset(0);
    this->ui->custom_plot_widget->xAxis->setPadding(0);
    this->ui->custom_plot_widget->xAxis->setAntialiased(true);

    this->ui->custom_plot_widget->yAxis->setLabel("probability");
    this->ui->custom_plot_widget->yAxis->setOffset(0);
    this->ui->custom_plot_widget->yAxis->setPadding(0);
    this->ui->custom_plot_widget->yAxis->setAntialiased(true);

    this->ui->custom_plot_widget->clearGraphs();
    QCPGraph* graph = this->ui->custom_plot_widget->addGraph();
    graph->setPen(QPen(QColor(116,205,122)));
    graph->setLineStyle(QCPGraph::lsStepCenter);
    graph->setErrorType(QCPGraph::etValue);

    QVector<double> intensitiesQ = QVector<double>::fromStdVector(intensities);
    QVector<double> probabilitiesQ = QVector<double>::fromStdVector(probabilities);
    graph->setData(intensitiesQ, probabilitiesQ);


    this->ui->custom_plot_widget->rescaleAxes();
    this->ui->custom_plot_widget->replot();
}

void HistogramWidget::on_histogram_bin_count_spinbox_valueChanged(int arg1)
{
    this->calculateHistogram();
}

void HistogramWidget::on_window_from_spinbox_valueChanged(double value)
{
    this->calculateHistogram();
    ITKToQImageConverter::setWindowFrom(value);
    emit fireImageRepaint();
}

void HistogramWidget::on_window_to_spinbox_valueChanged(double value)
{
    this->calculateHistogram();
    ITKToQImageConverter::setWindowTo(value);
    emit fireImageRepaint();
}

void HistogramWidget::on_fromMinimumButton_clicked()
{
    auto minimum = HistogramProcessor::minPixel(this->image);
    this->ui->window_from_spinbox->setValue(minimum);
    this->calculateHistogram();
}

void HistogramWidget::on_toMaximumButton_clicked()
{
    auto maximum = HistogramProcessor::maxPixel(this->image);
    this->ui->window_to_spinbox->setValue(maximum);
    this->calculateHistogram();
}
