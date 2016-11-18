#ifndef CHARTWIDGET_H
#define CHARTWIDGET_H

#include <QWidget>

namespace Ui {
class ChartWidget;
}

class ChartWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ChartWidget(QWidget *parent = 0);
    ~ChartWidget();

    void setAxisTitles(const QString xAxis, const QString yAxis);
    void setData(const QVector<double> xData, const QVector<double> yData);
    double getXAxisValue(int x);

    bool save(const QString file_name);
private:
    Ui::ChartWidget *ui;

signals:
    void chart_mouse_move(QMouseEvent* event);
};

#endif // CHARTWIDGET_H
