#include "HistogramWidget.h"
#include "ui_HistogramWidget.h"

#include "HistogramProcessor.h"
#include "ITKToQImageConverter.h"
#include "CudaImageOperationsProcessor.h"

#include <QClipboard>
#include <QFileDialog>

HistogramWidget::HistogramWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::HistogramWidget),
    image(ITKImage::Null)
{
    ui->setupUi(this);

    connect(this->ui->chart_widget, &ChartWidget::chart_mouse_move,
            this, &HistogramWidget::histogram_mouse_move);
    this->ui->chart_widget->setAxisTitles("intensity", "probability");

    qRegisterMetaType<std::vector<double>>("std::vector<double>");

    connect(this, &HistogramWidget::fireHistogramChanged,
            this, &HistogramWidget::handleHistogramChanged);
    connect(this, &HistogramWidget::fireEntropyLabelTextChange,
            this, &HistogramWidget::handleEntropyLabelTextChange);
    connect(this, &HistogramWidget::fireKernelBandwidthAndWindowChange,
            this, &HistogramWidget::handleKernelBandwidthAndWindowChange);
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
    double pixel_value = this->ui->chart_widget->getXAxisValue(position.x());

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

    // connect to mask changing...
    auto module = image_widget->getModuleByName("Mask");
    auto mask_module = dynamic_cast<MaskWidget*>(module);
    if(mask_module == nullptr)
        throw std::runtime_error("did not find mask module");
    connect(mask_module, &MaskWidget::maskChanged,
            this, [this](ITKImage) {
        if(this->ui->use_mask_module_checkbox->isChecked())
            this->calculateHistogram();
    });

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

ITKImage HistogramWidget::processImage(ITKImage)
{
    if(this->image.isNull())
        return ITKImage();

    ITKImage mask = this->ui->use_mask_module_checkbox->isChecked() ?
        mask_fetcher() : ITKImage();

    ITKImage::PixelType kernel_bandwidth = this->ui->kernel_bandwidth->value();
    ITKImage::PixelType window_from = this->ui->window_from_spinbox->value();
    ITKImage::PixelType window_to = this->ui->window_to_spinbox->value();

    if(this->ui->estimate_bandwidth_and_window_checkbox->isChecked())
        this->estimateBandwidthAndWindow(image, mask, window_from, window_to, kernel_bandwidth);

    uint spectrum_bandwidth = this->ui->spectrum_bandwidth_spinbox->value();
    HistogramProcessor::KernelType kernel_type =
            (this->ui->uniform_kernel_checkbox->isChecked() ? HistogramProcessor::Uniform :
            (this->ui->gaussian_kernel_checkbox->isChecked() ? HistogramProcessor::Gaussian :
            (this->ui->cosine_kernel_checkbox->isChecked() ? HistogramProcessor::Cosine :
             HistogramProcessor::Epanechnik)));

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
    auto intensitiesQ = QVector<double>::fromStdVector(intensities);
    auto probabilitiesQ = QVector<double>::fromStdVector(probabilities);
    this->ui->chart_widget->clearData();
    this->ui->chart_widget->addData(intensitiesQ, probabilitiesQ, "Histogram Series", QColor(116,205,122));
    this->ui->chart_widget->createDefaultAxes();

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

void HistogramWidget::calculateHistogramSync()
{
    processImage(this->image);
}

void HistogramWidget::on_window_from_spinbox_valueChanged(double value)
{
    if(!this->ui->window_from_spinbox->isEnabled())
        return;

    this->calculateHistogram();
    ITKToQImageConverter::setWindowFrom(value);
    emit fireImageRepaint();
}

void HistogramWidget::on_window_to_spinbox_valueChanged(double value)
{
    if(!this->ui->window_to_spinbox->isEnabled())
        return;

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
    if(!this->ui->kernel_bandwidth->isEnabled())
        return;

    this->calculateHistogram();
}

void HistogramWidget::on_uniform_kernel_checkbox_toggled(bool checked)
{
    if(checked)
        this->calculateHistogram();
}

void HistogramWidget::on_spectrum_bandwidth_spinbox_valueChanged(int)
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

    this->write(file_name);
}

void HistogramWidget::write(QString file_name)
{
    bool saved = this->ui->chart_widget->save(file_name);
    this->setStatusText( (saved ? "saved " : "(pdf,png supported) error while saving ") + file_name);
}

void HistogramWidget::on_use_mask_module_checkbox_clicked()
{
    this->calculateHistogram();
}

void HistogramWidget::on_estimate_bandwidth_and_window_checkbox_toggled(bool estimate_bandwidth_and_window)
{
    this->ui->kernel_bandwidth->setEnabled(!estimate_bandwidth_and_window);
    this->ui->window_from_spinbox->setEnabled(!estimate_bandwidth_and_window);
    this->ui->window_to_spinbox->setEnabled(!estimate_bandwidth_and_window);
    this->ui->fromMinimumButton->setEnabled(!estimate_bandwidth_and_window);
    this->ui->toMaximumButton->setEnabled(!estimate_bandwidth_and_window);

    this->calculateHistogram();
}

void HistogramWidget::handleKernelBandwidthAndWindowChange(double kernel_bandwidth,
                                                           double window_from,
                                                           double window_to)
{
    if(kernel_bandwidth != this->ui->kernel_bandwidth->value())
        this->ui->kernel_bandwidth->setValue(kernel_bandwidth);

    if(window_from != this->ui->window_from_spinbox->value())
        this->ui->window_from_spinbox->setValue(window_from);

    if(window_to != this->ui->window_to_spinbox->value())
        this->ui->window_to_spinbox->setValue(window_to);
}

void HistogramWidget::estimateBandwidthAndWindow(const ITKImage& image, const ITKImage& mask,
                                                 ITKImage::PixelType& window_from,
                                                 ITKImage::PixelType& window_to,
                                                 ITKImage::PixelType& kernel_bandwidth)
{
    const ITKImage::PixelType KDE_BANDWIDTH_FACTOR = 0.03;

    if(mask.isNull())
        image.minimumAndMaximum(window_from, window_to);
    else
        image.minimumAndMaximumInsideMask(window_from, window_to, mask);

    ITKImage::PixelType spectrum_range = window_to - window_from;
    kernel_bandwidth = spectrum_range * KDE_BANDWIDTH_FACTOR;

    emit this->fireKernelBandwidthAndWindowChange(kernel_bandwidth, window_from, window_to);
}

ITKImage::PixelType HistogramWidget::getEntropy()
{
    return this->ui->entropy_label->text().toDouble();
}
