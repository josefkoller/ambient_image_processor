#ifndef IMAGEVIEWWIDGET_H
#define IMAGEVIEWWIDGET_H

#include <QWidget>
#include <QMouseEvent>
#include <QWheelEvent>

#include "BaseModuleWidget.h"

namespace Ui {
class ImageViewWidget;
}

class CrosshairModule;

class ImageViewWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    ImageViewWidget(QString title, QWidget *parent);
    ~ImageViewWidget();

    virtual void registerModule(ImageWidget* image_widget);
    void registerCrosshairSubmodule(ImageWidget* image_widget);

    void setImage(ITKImage image);
    ITKImage getImage() const;

    void save_file_with_overlays();
private:
    Ui::ImageViewWidget *ui;

    CrosshairModule* crosshair_module;

    ITKImage image;
    QLabel* inner_image_frame;
    QImage* q_image;

    uint slice_index;

    void paintImage(bool repaint = false);

protected:
    void paintEvent(QPaintEvent *);
    void mousePressEvent(QMouseEvent *);
    void mouseReleaseEvent(QMouseEvent *);
    bool eventFilter(QObject *target, QEvent *event);

signals:
    void pixmapPainted(QPixmap* q_image);

    void mousePressedOnImage(Qt::MouseButton button, ITKImage::Index position);
    void mouseMoveOnImage(Qt::MouseButtons button, ITKImage::Index cursor_index);
    void mouseReleasedOnImage();
    void mouseWheelOnImage(int delta);

    void imageChanged(ITKImage image);
    void fireImageChange(ITKImage image);
public slots:
    void repaintImage();
    void repaintImageOverlays();
    void sliceIndexChanged(uint slice_index);

private slots:
    void handleImageChange(ITKImage image);
};

#endif // IMAGEVIEWWIDGET_H
