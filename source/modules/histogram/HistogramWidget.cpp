#include "HistogramWidget.h"
#include "ui_HistogramWidget.h"

#include "HistogramProcessor.h"

#include "ITKToQImageConverter.h"

HistogramWidget::HistogramWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::HistogramWidget),
    image(ITKImage::Null)
{
    ui->setupUi(this);

    this->ui->custom_plot_widget->setMouseTracking(true);
    connect(this->ui->custom_plot_widget, &QCustomPlot::mouseMove,
            this, &HistogramWidget::histogram_mouse_move);

    this->ui->custom_plot_widget->xAxis->setLabel("intensity");
    this->ui->custom_plot_widget->xAxis->setNumberPrecision(8);
    this->ui->custom_plot_widget->xAxis->setOffset(0);
    this->ui->custom_plot_widget->xAxis->setPadding(0);
    this->ui->custom_plot_widget->xAxis->setAntialiased(true);

    this->ui->custom_plot_widget->yAxis->setLabel("probability");
    this->ui->custom_plot_widget->yAxis->setOffset(0);
    this->ui->custom_plot_widget->yAxis->setPadding(0);
    this->ui->custom_plot_widget->yAxis->setAntialiased(true);

    qRegisterMetaType<std::vector<double>>("std::vector<double>");

    connect(this, &HistogramWidget::fireHistogramChanged,
            this, &HistogramWidget::handleHistogramChanged);
    connect(this, &HistogramWidget::fireEntropyLabelTextChange,
            this, &HistogramWidget::handleEntropyLabelTextChange);
}

HistogramWidget::~HistogramWidget()
{
    delete ui;
}

void HistogramWidget::histogram_mouse_move(QMouseEvent* event)
{
    if(this->image.isNull())
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
            image_widget, &ImageWidget::repaintImage);
}

bool HistogramWidget::calculatesResultImage() const
{
    return false;
}

void HistogramWidget::handleImageChanged(ITKImage image)
{
    this->image = image;
    this->calculateHistogram();
}

ITKImage HistogramWidget::processImage(ITKImage image)
{
    if(this->image.isNull())
        return ITKImage();

    uint spectrum_bandwidth = this->ui->spectrum_bandwidth_spinbox->value();
    ITKImage::PixelType kernel_bandwidth = this->ui->kernel_bandwidth->value();
    HistogramProcessor::KernelType kernel_type =
            (this->ui->uniform_kernel_checkbox->isChecked() ? HistogramProcessor::Uniform :
            (this->ui->gaussian_kernel_checkbox->isChecked() ? HistogramProcessor::Gaussian :
            (this->ui->cosine_kernel_checkbox->isChecked() ? HistogramProcessor::Cosine :
             HistogramProcessor::Epanechnik)));

    ITKImage::PixelType window_from = this->ui->window_from_spinbox->value();
    ITKImage::PixelType window_to = this->ui->window_to_spinbox->value();

    std::vector<double> intensities;
    std::vector<double> probabilities;
    HistogramProcessor::calculate(this->image,
                                  spectrum_bandwidth,
                                  kernel_bandwidth,
                                  kernel_type,
                                  window_from, window_to,
                                  intensities, probabilities);

    emit this->fireHistogramChanged(intensities, probabilities);

    return ITKImage();
}

void HistogramWidget::handleHistogramChanged(
        std::vector<double> intensities,
        std::vector<double> probabilities)
{
    this->ui->custom_plot_widget->clearGraphs();
    QCPGraph* graph = this->ui->custom_plot_widget->addGraph();
    graph->setPen(QPen(QColor(116,205,122)));
    graph->setLineStyle(QCPGraph::lsStepCenter);
    graph->setErrorType(QCPGraph::etValue);

    auto intensitiesQ = QVector<double>::fromStdVector(intensities);
    auto probabilitiesQ = QVector<double>::fromStdVector(probabilities);

    graph->setData(intensitiesQ, probabilitiesQ);

    this->ui->custom_plot_widget->rescaleAxes();
    this->ui->custom_plot_widget->replot();

    calculateEntropy(probabilities);
}

void HistogramWidget::calculateEntropy(const std::vector<double>& probabilities)
{
    auto entropy = HistogramProcessor::calculateEntropy(probabilities);
    emit this->fireEntropyLabelTextChange(QString::number(entropy));
}

void HistogramWidget::handleEntropyLabelTextChange(QString text)
{
    this->ui->entropy_label->setText(text);
}

void HistogramWidget::on_histogram_bin_count_spinbox_valueChanged(int arg1)
{
    this->calculateHistogram();
}

void HistogramWidget::calculateHistogram()
{
    if(!this->image.isNull())
        this->processInWorkerThread();
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
    this->ui->window_from_spinbox->setValue(this->image.minimum());
    this->calculateHistogram();
}

void HistogramWidget::on_toMaximumButton_clicked()
{
    this->ui->window_to_spinbox->setValue(this->image.maximum());
    this->calculateHistogram();
}

void HistogramWidget::on_kernel_bandwidth_valueChanged(double arg1)
{
    this->calculateHistogram();
}

void HistogramWidget::on_uniform_kernel_checkbox_toggled(bool checked)
{
    if(checked)
        this->calculateHistogram();
}

void HistogramWidget::on_spectrum_bandwidth_spinbox_valueChanged(int arg1)
{
    this->calculateHistogram();
}

void HistogramWidget::on_epanechnik_kernel_checkbox_toggled(bool checked)
{
    if(checked)
        this->calculateHistogram();
}

void HistogramWidget::on_cosine_kernel_checkbox_toggled(bool checked)
{
    if(checked)
        this->calculateHistogram();
}

void HistogramWidget::on_gaussian_kernel_checkbox_toggled(bool checked)
{
    if(checked)
        this->calculateHistogram();
}
