#ifndef MASKWIDGET_H
#define MASKWIDGET_H

#include "BaseModuleWidget.h"
#include "ImageViewWidget.h"

#include <functional>

namespace Ui {
class MaskWidget;
}

class MaskWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    MaskWidget(QString, QWidget *parent);
    ~MaskWidget();

    void registerModule(ImageWidget *image_widget);

    typedef std::function<ITKImage()> MaskFetcher;
    static MaskFetcher createMaskFetcher(ImageWidget* image_widget);

    void setMaskImage(ITKImage mask);
private slots:
    void on_load_mask_button_clicked();
    void on_enabled_checkbox_clicked();

private:
    Ui::MaskWidget *ui;
    ImageViewWidget* mask_view;

    ITKImage getMask() const;

signals:
    void maskChanged(ITKImage mask);
};

#endif // MASKWIDGET_H
