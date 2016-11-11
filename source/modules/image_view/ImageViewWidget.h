#ifndef IMAGEVIEWWIDGET_H
#define IMAGEVIEWWIDGET_H

#include <QWidget>
#include <QMouseEvent>
#include <QWheelEvent>

#include "BaseModuleWidget.h"

namespace Ui {
class ImageViewWidget;
}

typedef std::function<ITKImage()> MaskFetcher;

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
    void load_color_to_view_only_clicked();
private:
    Ui::ImageViewWidget *ui;

    CrosshairModule* crosshair_module;

    ITKImage image;
    QImage* q_image;

    uint slice_index;

    bool do_rescale;
    bool do_multiply;
    bool use_window;
    bool use_mask_module;

    MaskFetcher mask_fetcher;

    void paintImage(bool repaint = false);

protected:
    void paintEvent(QPaintEvent *);
    void mousePressEvent(QMouseEvent *);
    void mouseReleaseEvent(QMouseEvent *);
    void wheelEvent(QWheelEvent *);

    bool eventFilter(QObject *watched, QEvent *event);

    void setBorder(bool enabled, QColor color);
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

    void doRescaleChanged(bool do_rescale);
    void doMultiplyChanged(bool do_multiply);
    void useWindowChanged(bool use_window);
    void useMaskModule(bool use_mask_module);
private slots:
    void handleImageChange(ITKImage image);
};

#endif // IMAGEVIEWWIDGET_H
