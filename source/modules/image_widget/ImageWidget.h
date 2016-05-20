#ifndef IMAGEWIDGET_H
#define IMAGEWIDGET_H

#include <QWidget>
#include <QList>
#include <QLabel>

#include "ITKImageProcessor.h"
#include "ITKToQImageConverter.h"

#include <functional>

#include <QListWidgetItem>

#include "ITKImage.h"

class BaseModule;

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


    void showSliceControl();

    void showImageOnly();

    void connectSliceControlTo(ImageWidget* other_image_widget);

    void connectModule(QString module_title, ImageWidget* other_image_widget);

private:
    BaseModule* getModuleByName(QString module_title) const;
private slots:
    void on_slice_slider_valueChanged(int value);

    void on_load_button_clicked();

    void on_save_button_clicked();

public:
    Ui::ImageWidget *ui;
private:
    QList<BaseModule*> modules;

    ImageWidget* output_widget;
    ImageWidget* output_widget2;
    ImageWidget* output_widget3;

    ITKImage image;

    bool show_slice_control;

    void setSliceIndex(uint slice_index);
    uint userSliceIndex() const;

    void paintImage(bool repaint = false);
    void setInputRanges();
signals:
    void sliceIndexChanged(uint slice_index);
public slots:
    void connectedSliceControlChanged(uint slice_index);

public:
    ITKImage getImage() { return this->image; }
private slots:
    void on_slice_spinbox_valueChanged(int arg1);

public:
    void setMinimumSizeToImage();
    void hidePixelValueAtCursor();
    void showPixelValueAtCursor();
    void setOutputWidget(ImageWidget* output_widget);
    void setOutputWidget2(ImageWidget* output_widget);
    void setOutputWidget3(ImageWidget* output_widget);

    ImageWidget* getOutputWidget() const;

    void setPage(unsigned char page_index);
protected:
    void paintEvent(QPaintEvent *);
    void mousePressEvent(QMouseEvent *);
    void mouseReleaseEvent(QMouseEvent *);

signals:
    void fireStatusTextChange(QString text);
    void fireImageChange(ITKImage image);
    void imageChanged(ITKImage image);
    void pixmapPainted(QPixmap* q_image);
    void mousePressedOnImage(Qt::MouseButton button, QPoint position);
    void mouseMoveOnImage(Qt::MouseButtons button, QPoint position);
    void mouseReleasedOnImage();

private slots:
    void handleStatusTextChange(QString text);
    void handleImageChange(ITKImage image);

private:
    QLabel* inner_image_frame;
    QImage* q_image;
protected:
    bool eventFilter(QObject *target, QEvent *event);

public slots:
    void handleRepaintImage();
};

#endif // IMAGEWIDGET_H
