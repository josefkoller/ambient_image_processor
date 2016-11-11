#include "HistogramWidget.h"
#include "ui_HistogramWidget.h"

#include "HistogramProcessor.h"

#include "ITKToQImageConverter.h"

#include <QClipboard>
#include <QFileDialog>

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

    this->mask_fetcher = MaskWidget::createMaskFetcher(image_widget);
}

bool HistogramWidget::calculatesResultImage() const
{
    return false;
}

void HistogramWidget::handleImageChanged(ITKImage image)
{
    this->image = image;

    if(this->ui->refresh_on_image_change->isChecked())
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

    ITKImage mask = this->ui->use_mask_module_checkbox->isChecked() ?
        mask_fetcher() : ITKImage();

    std::vector<double> intensities;
    std::vector<double> probabilities;
    HistogramProcessor::calculate(this->image,
                                  mask,
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

void HistogramWidget::on_copy_to_clipboard_button_clicked()
{
    auto text =
            this->ui->entropy_group_box->title() + ": " +
            this->ui->entropy_label->text();

    QApplication::clipboard()->setText(text);
}

void HistogramWidget::on_save_button_clicked()
{
    if(this->image.isNull())
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save histogram image file");
    if(file_name.isNull())
        return;

    bool saved = false;
    if(file_name.endsWith("pdf"))
        saved = this->ui->custom_plot_widget->savePdf(file_name);
    if(file_name.endsWith("png"))
        saved = this->ui->custom_plot_widget->savePng(file_name,0,0,1.0, 100);  // 100 ... uncompressed

    this->setStatusText( (saved ? "saved " : "(pdf,png supported) error while saving ") + file_name);
}

void HistogramWidget::on_use_mask_module_checkbox_clicked()
{
    this->calculateHistogram();
}
