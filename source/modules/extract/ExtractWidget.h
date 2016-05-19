#ifndef ExtractWIDGET_H
#define ExtractWIDGET_H

#include <QWidget>

#include "BaseModuleWidget.h"

namespace Ui {
class ExtractWidget;
}

class ExtractWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit ExtractWidget(QWidget *parent = 0);
    ~ExtractWidget();
private slots:
    void on_extract_button_clicked();

    void imageChanged(ITKImage::InnerITKImage::Pointer image);
    void updateExtractedSizeLabel(int);
private:
    Ui::ExtractWidget *ui;

protected:
    ITKImage processImage(ITKImage image);

public:
    void registerModule(ImageWidget* image_widget);
};

#endif // ExtractWIDGET_H
