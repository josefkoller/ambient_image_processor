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

    void save_file_with_overlays(QString file_name = "");
    void load_color_to_view_only(QString file_name = "");
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
