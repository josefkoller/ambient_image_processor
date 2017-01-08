/*
    Ambient Image Processor - A tool to perform several imaging tasks
    
    Copyright (C) 2016 Josef Koller

    https://github.com/josefkoller/ambient_image_processor    
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

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
class ImageViewWidget;

namespace Ui {
class ImageWidget;
}

Q_DECLARE_METATYPE(ITKImage);
Q_DECLARE_METATYPE(ITKImage::Index);

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

    BaseModule* getModuleByName(QString module_title) const;
private:
    Ui::ImageWidget *ui;
    QList<BaseModule*> modules;

    ImageWidget* output_widget;
    ImageWidget* output_widget2;
    ImageWidget* output_widget3;

    ITKImage image;

    void paintImage(bool repaint = false);
    void setMinimumSizeToImage();

    ImageViewWidget* image_view_widget;
    SliceControlWidget* slice_control_widget;

    QMenu *image_menu;

    void on_load_button_clicked();
    void on_save_button_clicked();
    void load_hsv_clicked();
    void save_hsv_clicked();

signals:
    void fireStatusTextChange(QString text);
    void fireImageChange(ITKImage image);
    void imageChanged(ITKImage image);

    void mousePressedOnImage(Qt::MouseButton button, ITKImage::Index position);
    void mouseMoveOnImage(Qt::MouseButtons button, ITKImage::Index cursor_index);
    void mouseReleasedOnImage();
    void mouseWheelOnImage(int delta);

    void pixmapPainted(QPixmap* q_image);
    void sliceIndexChanged(uint slice_index);

    void repaintImage();
    void repaintImageOverlays();
private slots:
    void handleStatusTextChange(QString text);
    void handleImageChange(ITKImage image);
};

#endif // IMAGEWIDGET_H
