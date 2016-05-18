#ifndef NONLOCALGRADIENTKERNELWIDGET_H
#define NONLOCALGRADIENTKERNELWIDGET_H

#include <QWidget>
#include <QPaintEvent>

namespace Ui {
class NonLocalGradientKernelWidget;
}

class NonLocalGradientKernelWidget : public QWidget
{
    Q_OBJECT

public:
    explicit NonLocalGradientKernelWidget(QWidget *parent = 0);
    ~NonLocalGradientKernelWidget();

    void setSigma(float sigma);
    void setKernelSize(uint kernel_size);
private:
    Ui::NonLocalGradientKernelWidget *ui;
    float sigma;
    uint kernel_size;
    QImage kernel_image;

    void createKernelImage();
protected:
    void paintEvent(QPaintEvent *);
};

#endif // NONLOCALGRADIENTKERNELWIDGET_H
