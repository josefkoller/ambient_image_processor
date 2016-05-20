#ifndef IMAGEWIDGET_H
#define IMAGEWIDGET_H

#include <QWidget>
#include <QList>
#include <QLabel>

#include "ITKToQImageConverter.h"

#include <functional>

#include <QListWidgetItem>

#include "ITKImage.h"

class BaseModule;
class SliceControlWidget;

namespace Ui {
class ImageWidget;
}

Q_DECLARE_METATYPE(ITKImage);

class ImageWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ImageWidget(QWidget *parent = 0);
    ~ImageWidget();

    void setImage(ITKImage image);
    ITKImage getImage() { return this->image; }

    void showImageOnly();
    void connectModule(QString module_title, ImageWidget* other_image_widget);

    void setOutputWidget(ImageWidget* output_widget);
    void setOutputWidget2(ImageWidget* output_widget);
    void setOutputWidget3(ImageWidget* output_widget);
    ImageWidget* getOutputWidget() const;
    void setPage(unsigned char page_index);

private:
    Ui::ImageWidget *ui;
    QList<BaseModule*> modules;

    ImageWidget* output_widget;
    ImageWidget* output_widget2;
    ImageWidget* output_widget3;

    ITKImage image;

    QLabel* inner_image_frame;
    QImage* q_image;

    void paintImage(bool repaint = false);
    void setMinimumSizeToImage();
    BaseModule* getModuleByName(QString module_title) const;

    SliceControlWidget* slice_control_widget;
private slots:
    void on_load_button_clicked();
    void on_save_button_clicked();

protected:
    void paintEvent(QPaintEvent *);
    void mousePressEvent(QMouseEvent *);
    void mouseReleaseEvent(QMouseEvent *);
    bool eventFilter(QObject *target, QEvent *event);

signals:
    void fireStatusTextChange(QString text);
    void fireImageChange(ITKImage image);
    void imageChanged(ITKImage& image);
    void pixmapPainted(QPixmap* q_image);
    void mousePressedOnImage(Qt::MouseButton button, ITKImage::Index position);
    void mouseMoveOnImage(Qt::MouseButtons button, ITKImage::Index cursor_index);
    void mouseReleasedOnImage();
    void mouseWheelOnImage(int delta);

private slots:
    void handleStatusTextChange(QString text);
    void handleImageChange(ITKImage image);

public slots:
    void handleRepaintImage();
};

#endif // IMAGEWIDGET_H
