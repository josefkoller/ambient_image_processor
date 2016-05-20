#ifndef SPLINEINTERPOLATIONWIDGET_H
#define SPLINEINTERPOLATIONWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"


#include "SplineInterpolationProcessor.h"

namespace Ui {
class SplineInterpolationWidget;
}

class SplineInterpolationWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit SplineInterpolationWidget(QString title, QWidget *parent = 0);
    ~SplineInterpolationWidget();

private:
    Ui::SplineInterpolationWidget *ui;

    ITKImage::InnerITKImage::Pointer image;
    bool adding_reference_roi;
    QList<QVector<QPoint>> reference_rois;
    std::vector<SplineInterpolationProcessor::ReferenceROIStatistic> reference_rois_statistic;

    int selectedReferenceROI();
    void paintSelectedReferenceROI(QPixmap* pixmap);
    void updateReferenceROI();

    void setReferenceROIs(QList<QVector<QPoint>> reference_rois);
protected:
    ITKImage processImage(ITKImage image);
private slots:
    void on_pushButton_6_clicked();
    void on_add_reference_roi_button_clicked();
    void mouseMoveOnImage(Qt::MouseButtons button, QPoint position);
    void mouseReleasedOnImage();
    void on_referenceROIsListWidget_currentRowChanged(int currentRow);

    void on_referenceROIsListWidget_itemSelectionChanged();

public:
    void registerModule(ImageWidget* image_widget);

signals:
    void repaintImage();
};

#endif // SPLINEINTERPOLATIONWIDGET_H
