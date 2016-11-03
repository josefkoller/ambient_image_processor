#ifndef MANUALDRAWMASK_H
#define MANUALDRAWMASK_H

#include "BaseModuleWidget.h"

namespace Ui {
class ManualDrawMask;
}

class ManualDrawMask : public BaseModuleWidget
{
    Q_OBJECT

public:
    ManualDrawMask(QString title, QWidget *parent = 0);
    ~ManualDrawMask();

protected:
    ITKImage processImage(ITKImage image);
    void registerModule(ImageWidget *image_widget);

private:
    Ui::ManualDrawMask *ui;

    bool is_drawing_mask;
    QVector<ITKImage::Index> polygon_points;
    const Qt::FillRule polygon_fill_rule;

    QPolygon createPolygon();
private slots:
    void mouseMoveOnImage(Qt::MouseButtons buttons, ITKImage::Index cursor_index);
    void mouseReleasedOnImage();
    void paintPolygon(QPixmap *pixmap);

    void on_startButton_clicked();

signals:
    void repaintImage();
};

#endif // MANUALDRAWMASK_H
