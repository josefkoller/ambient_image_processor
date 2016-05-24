#ifndef IMAGEVIEWWIDGET_H
#define IMAGEVIEWWIDGET_H

#include <QWidget>
#include <QMouseEvent>
#include <QWheelEvent>

#include "BaseModuleWidget.h"

namespace Ui {
class ImageViewWidget;
}

class ImageViewWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    ImageViewWidget(QString title, QWidget *parent);
    ~ImageViewWidget();

    virtual void registerModule(ImageWidget* image_widget);
private:
    Ui::ImageViewWidget *ui;

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

public slots:
    void repaintImage();
};

#endif // IMAGEVIEWWIDGET_H
