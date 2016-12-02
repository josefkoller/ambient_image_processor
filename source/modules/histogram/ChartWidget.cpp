#include "ChartWidget.h"

#include <QHBoxLayout>

#include <QtCharts/QChart>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>

#include <QPdfWriter>
#include <QImageWriter>

#include <iostream>

ChartWidget::ChartWidget(QWidget *parent) :
    QWidget(parent),
    xAxisTitle("x"),
    yAxisTitle("y")
{
    QtCharts::QChart *chart = new QtCharts::QChart();
    chart->legend()->hide();
    chart->legend()->setAlignment(Qt::AlignBottom);

    this->chart_view = new MouseHandlingChartView(chart, this);
    this->chart_view->setRenderHint(QPainter::Antialiasing);

    connect(this->chart_view, &MouseHandlingChartView::mouseMove,
            this, &ChartWidget::chart_mouse_move);

    this->setLayout(new QHBoxLayout());
    this->layout()->addWidget(this->chart_view);
    this->chart_view->setMinimumSize(360, 320); // for file writing without a gui
    this->chart_view->setBackgroundBrush(QBrush(Qt::white));
    this->chart_view->setContentsMargins(0,0,0,0);
}

ChartWidget::~ChartWidget()
{
}

void ChartWidget::setAxisTitles(const QString xAxisTitle, const QString yAxisTitle)
{
    this->xAxisTitle = xAxisTitle;
    this->yAxisTitle = yAxisTitle;
}

double ChartWidget::getXAxisValue(int x)
{
    return this->chart_view->chart()->mapToValue(QPointF(x,0)).x();
}

double ChartWidget::getYAxisValue(int y)
{
    return this->chart_view->chart()->mapToValue(QPointF(0,y)).y();
}

void ChartWidget::clearData()
{
    this->chart_view->chart()->removeAllSeries();
}

void ChartWidget::addData(const QVector<double> xData,
                          const QVector<double> yData,
                          QString series_title,
                          QPen series_pen) {
    QtCharts::QChart* chart = this->chart_view->chart();

    QtCharts::QLineSeries* line_series = new QtCharts::QLineSeries();

    for(int i = 0; i < xData.size(); i++) {
        line_series->append(xData[i], yData[i]);
    }
    line_series->setName(series_title);
    line_series->setPen(series_pen);

    chart->addSeries(line_series);

    /*
    if(chart->series().length() > 1)
        chart->legend()->show();
    */
}

void ChartWidget::addPoint(const double xData,
                          const double yData,
                          QString series_title,
                          QColor series_color) {
    QtCharts::QScatterSeries* series = new QtCharts::QScatterSeries();
    series->setName(series_title);
    series->setMarkerShape(QtCharts::QScatterSeries::MarkerShapeCircle);
    series->setMarkerSize(6);
    series->setPen(QPen(series_color));
    series->setBrush(QBrush(series_color));
    series->append(xData, yData);
    this->chart_view->chart()->addSeries(series);
}

void ChartWidget::createDefaultAxes()
{
    QtCharts::QChart* chart = this->chart_view->chart();
    chart->createDefaultAxes();
    chart->axisX()->setTitleText(this->xAxisTitle);
    chart->axisY()->setTitleText(this->yAxisTitle);
}

bool ChartWidget::save(const QString file_name)
{
    if(file_name.endsWith("pdf")) {
        QPdfWriter writer(file_name);
        writer.setCreator("ambient_image_processor");
        QPainter painter(&writer);
        this->chart_view->render(&painter);
        return painter.end();
    } else {
        auto pixmap = this->chart_view->grab();
        const int margin = 24;
        pixmap = pixmap.copy(margin, margin,
                             pixmap.rect().width() - margin*2,
                             pixmap.rect().height() - margin*2); // crop

        const char* format = 0; // choose by the filename
        const int quality = 100; // best quality
        return pixmap.save(file_name, format, quality);
    }

    return false;
}
