#ifndef REGIONCURVATUREEDGECORRECTION_H
#define REGIONCURVATUREEDGECORRECTION_H

#include <QWidget>

#include "BaseModuleWidget.h"

namespace Ui {
class RegionCurvatureEdgeCorrection;
}

class RegionCurvatureEdgeCorrection : public BaseModuleWidget
{
    Q_OBJECT

public:
    RegionCurvatureEdgeCorrection(QString title, QWidget *parent);
    ~RegionCurvatureEdgeCorrection();

    void registerModule(ImageWidget *image_widget);
private slots:
    void on_set_seed_point_button_clicked();
    void mousePressedOnImage(Qt::MouseButton button, ITKImage::Index position);
    void on_perform_button_clicked();

private:
    Ui::RegionCurvatureEdgeCorrection *ui;

    ITKImage::Index seed_point;

    void setSeedPoint(ITKImage::Index position);

protected:
    ITKImage processImage(ITKImage image);
};

#endif // REGIONCURVATUREEDGECORRECTION_H
