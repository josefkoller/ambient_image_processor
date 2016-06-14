#include "TGVDeshadeMetricPlotWidget.h"
#include "ui_TGVDeshadeMetricPlotWidget.h"

#include <QFileDialog>

TGVDeshadeMetricPlotWidget::TGVDeshadeMetricPlotWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::TGVDeshadeMetricPlotWidget)
{
    ui->setupUi(this);

    this->shading_output_view = new ImageViewWidget("Denoised", this->ui->shading_frame);
    this->ui->shading_frame->layout()->addWidget(this->shading_output_view);

    this->denoised_output_view = new ImageViewWidget("Denoised", this->ui->denoised_frame);
    this->ui->denoised_frame->layout()->addWidget(this->denoised_output_view);

    this->mask_view = new ImageViewWidget("Mask", this->ui->mask_frame);
    this->ui->mask_frame->layout()->addWidget(this->mask_view);

    qRegisterMetaType<std::vector<double>>("std::vector<double>");

    connect(this, &TGVDeshadeMetricPlotWidget::fireMetricValuesChanged,
            this, &TGVDeshadeMetricPlotWidget::handleMetricValuesChanged);


    this->ui->metric_plot->xAxis->setLabel("iteration");
    this->ui->metric_plot->xAxis->setOffset(0);
    this->ui->metric_plot->xAxis->setPadding(0);
    this->ui->metric_plot->xAxis->setAntialiased(true);

    this->ui->metric_plot->yAxis->setLabel("metric");
    this->ui->metric_plot->xAxis->setNumberPrecision(8);
    this->ui->metric_plot->yAxis->setOffset(0);
    this->ui->metric_plot->yAxis->setPadding(0);
    this->ui->metric_plot->yAxis->setAntialiased(true);
}

TGVDeshadeMetricPlotWidget::~TGVDeshadeMetricPlotWidget()
{
    delete ui;
}

void TGVDeshadeMetricPlotWidget::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(this, &BaseModuleWidget::fireWorkerFinished,
            this, [this]() {
        this->ui->stop_button->setEnabled(false);
    });

    this->shading_output_view->registerCrosshairSubmodule(image_widget);
    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this->shading_output_view, &ImageViewWidget::sliceIndexChanged);
    this->denoised_output_view->registerCrosshairSubmodule(image_widget);
    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this->denoised_output_view, &ImageViewWidget::sliceIndexChanged);
}

void TGVDeshadeMetricPlotWidget::setIterationFinishedCallback(TGVDeshadeProcessor::IterationFinished iteration_finished_callback)
{
    this->iteration_finished_callback = [this, iteration_finished_callback]
            (uint iteration_index, uint iteration_count,
            TGVDeshadeMetricPlotProcessor::MetricValues metricValues,
            ITKImage u, ITKImage l, ITKImage r){

        iteration_finished_callback(iteration_index, iteration_count, r);

        emit this->shading_output_view->fireImageChange(l);
        emit this->denoised_output_view->fireImageChange(u);
        emit this->fireMetricValuesChanged(metricValues);

        return this->stop_after_next_iteration;
    };
}

void TGVDeshadeMetricPlotWidget::handleMetricValuesChanged(std::vector<double> metricValues)
{
    this->plotMetricValues(metricValues);
}

ITKImage TGVDeshadeMetricPlotWidget::processImage(ITKImage image)
{
    const float alpha0 = this->ui->alpha0_spinbox->value();
    const float alpha1 = this->ui->alpha1_spinbox->value();
    const float lambda = this->ui->lambda_spinbox->value();
    const uint iteration_count = this->ui->iteration_count_spinbox->value();
    const uint paint_iteration_interval = iteration_count + 1; // do not paint in between

    auto mask = this->mask_view->getImage();

    const auto metric_type = this->ui->normalized_cross_correlation_checkbox->isChecked() ?
                TGVDeshadeMetricPlotProcessor::NormalizedCrossCorrelation :
                TGVDeshadeMetricPlotProcessor::SumOfAbsoluteDifferences;

    QString metricName = this->ui->normalized_cross_correlation_checkbox->isChecked() ?
                this->ui->normalized_cross_correlation_checkbox->text() :
                this->ui->sum_of_absolute_differences_checkbox->text();
    this->ui->metric_plot->yAxis->setLabel(metricName);

    ITKImage denoised_image = ITKImage();
    ITKImage shading_image = ITKImage();
    ITKImage deshaded_image = ITKImage();

    auto metricValues = TGVDeshadeMetricPlotProcessor::processTGV2L1DeshadeCuda(
                image,
                lambda,
                alpha0,
                alpha1,
                iteration_count,
                mask,
                metric_type,

                paint_iteration_interval,
                this->iteration_finished_callback,

                denoised_image,
                shading_image,
                deshaded_image);

    emit this->fireMetricValuesChanged(metricValues);

    emit this->denoised_output_view->fireImageChange(denoised_image);
    emit this->shading_output_view->fireImageChange(shading_image);
    return deshaded_image;
}

void TGVDeshadeMetricPlotWidget::on_load_mask_button_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "open volume file");
    if(file_name == QString::null || !QFile(file_name).exists())
        return;

    this->mask_view->setImage(ITKImage::read(file_name.toStdString()));
}

void TGVDeshadeMetricPlotWidget::on_save_denoised_button_clicked()
{
    auto image = this->denoised_output_view->getImage();
    if(image.isNull())
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save volume file");
    if(file_name.isNull())
        return;

    image.write(file_name.toStdString());
}

void TGVDeshadeMetricPlotWidget::on_save_second_output_button_clicked()
{
    auto image = this->shading_output_view->getImage();
    if(image.isNull())
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save volume file");
    if(file_name.isNull())
        return;

    image.write(file_name.toStdString());
}

void TGVDeshadeMetricPlotWidget::on_stop_button_clicked()
{
    this->stop_after_next_iteration = true;
}

void TGVDeshadeMetricPlotWidget::on_perform_button_clicked()
{
    this->stop_after_next_iteration = false;
    this->processInWorkerThread();
    this->ui->stop_button->setEnabled(true);
}

void TGVDeshadeMetricPlotWidget::plotMetricValues(TGVDeshadeMetricPlotProcessor::MetricValues metricValues)
{
    this->ui->metric_plot->clearGraphs();
    QCPGraph *graph = this->ui->metric_plot->addGraph();

    double metricSum = 0;
    auto iterationsQ = QVector<double>();
    for(int i = 0; i < metricValues.size(); i++)
    {
        metricSum+= metricValues[i];
        iterationsQ.push_back(i);
    }
    std::cout << "metric mean : " << (metricSum / metricValues.size()) << std::endl;
    auto metricValuesQ = QVector<double>::fromStdVector(metricValues);
    graph->setData(iterationsQ, metricValuesQ);

    auto pen_color = QColor(0, 51, 153);
    graph->setPen(QPen(pen_color));
    graph->setLineStyle(QCPGraph::lsLine);
    graph->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 3));

    this->ui->metric_plot->rescaleAxes();
    this->ui->metric_plot->replot();
}

void TGVDeshadeMetricPlotWidget::on_save_metric_plot_button_clicked()
{
    QString file_name = QFileDialog::getSaveFileName(this, "save metric plot");
    if(file_name.isNull())
        return;

    bool saved = false;
    if(file_name.endsWith("pdf"))
        saved = this->ui->metric_plot->savePdf(file_name);
    if(file_name.endsWith("png"))
        saved = this->ui->metric_plot->savePng(file_name,0,0,1.0, 100);  // 100 ... uncompressed

    this->setStatusText( (saved ? "saved " : "(pdf,png supported) error while saving ") + file_name);
}
