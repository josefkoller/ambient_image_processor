#include "ChartWidget.h"
#include "ui_ChartWidget.h"

/*
#include <QChart>
#include <QBarSet>
#include <QLineSeries>
*/

ChartWidget::ChartWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ChartWidget)
{
    ui->setupUi(this);

    this->ui->custom_plot_widget->setMouseTracking(true);
    connect(this->ui->custom_plot_widget, &QCustomPlot::mouseMove,
            this, &ChartWidget::chart_mouse_move);

    this->ui->custom_plot_widget->xAxis->setNumberPrecision(8);
    this->ui->custom_plot_widget->xAxis->setOffset(0);
    this->ui->custom_plot_widget->xAxis->setPadding(0);
    this->ui->custom_plot_widget->xAxis->setAntialiased(true);

    this->ui->custom_plot_widget->yAxis->setOffset(0);
    this->ui->custom_plot_widget->yAxis->setPadding(0);
    this->ui->custom_plot_widget->yAxis->setAntialiased(true);

}

ChartWidget::~ChartWidget()
{
    delete ui;
}

void ChartWidget::setAxisTitles(const QString xAxis, const QString yAxis)
{
    this->ui->custom_plot_widget->xAxis->setLabel(xAxis);
    this->ui->custom_plot_widget->yAxis->setLabel(yAxis);
}

double ChartWidget::getXAxisValue(int x)
{
    return this->ui->custom_plot_widget->xAxis->pixelToCoord(x);
}

void ChartWidget::setData(const QVector<double> xData,
                          const QVector<double> yData) {
    this->ui->custom_plot_widget->clearGraphs();
    QCPGraph* graph = this->ui->custom_plot_widget->addGraph();
    graph->setPen(QPen(QColor(116,205,122)));
    graph->setLineStyle(QCPGraph::lsStepCenter);
    graph->setErrorType(QCPGraph::etValue);

    graph->setData(xData, yData);

    this->ui->custom_plot_widget->rescaleAxes();
    this->ui->custom_plot_widget->replot();
}

bool ChartWidget::save(const QString file_name)
{
    if(file_name.endsWith("pdf"))
        return this->ui->custom_plot_widget->savePdf(file_name);
    if(file_name.endsWith("png"))
        return this->ui->custom_plot_widget->savePng(file_name,0,0,1.0, 100);  // 100 ... uncompressed

    return false;
}

