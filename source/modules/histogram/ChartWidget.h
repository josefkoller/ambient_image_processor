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

#ifndef CHARTWIDGET_H
#define CHARTWIDGET_H

#include <QWidget>

#include <QtCharts/QChartView>
#include <QtCharts/QValueAxis>

class MouseHandlingChartView : public QtCharts::QChartView {
    Q_OBJECT
public:
    MouseHandlingChartView(QtCharts::QChart *chart, QWidget *parent) :
        QtCharts::QChartView(chart, parent) {
        this->setMouseTracking(true);
    }

signals:
    void mouseMove(QMouseEvent* event);

protected:
    void mouseMoveEvent(QMouseEvent* event) {
        emit this->mouseMove(event);
    }
};

class ChartWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ChartWidget(QWidget *parent = 0);
    ~ChartWidget();

    void setAxisTitles(const QString xAxisTitle, const QString yAxisTitle);
    void clearData();
    void addData(const QVector<double> xData, const QVector<double> yData,
                 QString series_title,
                 QPen series_pen);
    void createDefaultAxes();
    double getXAxisValue(int x);
    double getYAxisValue(int y);

    bool save(const QString file_name);
    void addPoint(const double xData, const double yData, QString series_title, QColor series_color);
private:
    MouseHandlingChartView* chart_view;

    QString xAxisTitle, yAxisTitle;
signals:
    void chart_mouse_move(QMouseEvent* position);
};

#endif // CHARTWIDGET_H
