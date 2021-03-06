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

#include "ImageViewWidget.h"
#include "ui_ImageViewWidget.h"

#include "ITKToQImageConverter.h"
#include "CrosshairModule.h"
#include "MaskWidget.h"

#include <QVBoxLayout>
#include <QFileDialog>

ImageViewWidget::ImageViewWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::ImageViewWidget),
    q_image(nullptr),
    image(ITKImage::Null),
    slice_index(0),
    do_rescale(true),
    do_multiply(false),
    use_window(false),
    use_mask_module(false)
{
    this->ui->setupUi(this);

    this->ui->image_frame->installEventFilter(this);

    connect(this, &ImageViewWidget::fireImageChange,
            this, &ImageViewWidget::handleImageChange);

    this->crosshair_module = new CrosshairModule(title + " Crosshair");
}

ImageViewWidget::~ImageViewWidget()
{
    delete ui;
}

void ImageViewWidget::registerModule(ImageWidget* image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(image_widget, &ImageWidget::imageChanged,
            this, [this](ITKImage image) {
        this->image = image;

        if(slice_index >= this->image.depth)
            this->slice_index = 0;

        this->repaintImage();
        emit this->imageChanged(image);
    });
    connect(image_widget, &ImageWidget::sliceIndexChanged,
            this, &ImageViewWidget::sliceIndexChanged);
    connect(this, &ImageViewWidget::pixmapPainted,
            image_widget, &ImageWidget::pixmapPainted);
    connect(this, &ImageViewWidget::mousePressedOnImage,
            image_widget, &ImageWidget::mousePressedOnImage);
    connect(this, &ImageViewWidget::mouseMoveOnImage,
            image_widget, &ImageWidget::mouseMoveOnImage);
    connect(this, &ImageViewWidget::mouseReleasedOnImage,
            image_widget, &ImageWidget::mouseReleasedOnImage);
    connect(this, &ImageViewWidget::mouseWheelOnImage,
            image_widget, &ImageWidget::mouseWheelOnImage);

    connect(image_widget, &ImageWidget::repaintImage,
            this, &ImageViewWidget::repaintImage);
    connect(image_widget, &ImageWidget::repaintImageOverlays,
            this, &ImageViewWidget::repaintImageOverlays);

    this->registerCrosshairSubmodule(image_widget);


    this->mask_fetcher = MaskWidget::createMaskFetcher(image_widget);

    // connect to mask changing...
    auto module = image_widget->getModuleByName("Mask");
    auto mask_module = dynamic_cast<MaskWidget*>(module);
    if(mask_module == nullptr)
        throw std::runtime_error("did not find mask module");
    connect(mask_module, &MaskWidget::maskChanged,
            this, [this](ITKImage) {
        if(!this->use_mask_module)
            return;

        this->repaintImage();
        emit this->imageChanged(image);
    });
}

void ImageViewWidget::registerCrosshairSubmodule(ImageWidget* image_widget)
{
    this->crosshair_module->registerModule(this, image_widget);
}

void ImageViewWidget::sliceIndexChanged(uint slice_index)
{
    this->slice_index = slice_index;
    this->repaintImage();
}

void ImageViewWidget::doRescaleChanged(bool do_rescale)
{
    this->do_rescale = do_rescale;
    this->repaintImage();
}

void ImageViewWidget::doMultiplyChanged(bool do_multiply)
{
    this->do_multiply = do_multiply;
    this->repaintImage();
}

void ImageViewWidget::useWindowChanged(bool use_window)
{
    this->use_window = use_window;

    this->setBorder(use_window, ITKToQImageConverter::upper_window_color);
    this->repaintImage();
}

void ImageViewWidget::setBorder(bool enabled, QColor color)
{
    QString color_text = QString("rgb(%1, %2, %3)").arg(
                QString::number(color.red()),
                QString::number(color.green()),
                QString::number(color.blue()));

    QString border_style = enabled ? "border: 2px solid " + color_text : "border: none";
    this->ui->image_frame->setStyleSheet(border_style);
}

void ImageViewWidget::useMaskModule(bool use_mask_module)
{
    this->use_mask_module = use_mask_module;

    this->setBorder(use_mask_module, ITKToQImageConverter::outside_mask_color);
    this->repaintImage();
}

void ImageViewWidget::paintImage(bool repaint)
{
    if(this->image.isNull())
    {
        ui->image_frame->setPixmap(QPixmap());
        return;
    }

    if(repaint && q_image != nullptr) {
        delete q_image;
        q_image = nullptr;
    }

    if(q_image != nullptr)
        return;

    if(this->slice_index > this->image.depth - 1)
        this->slice_index = this->image.depth - 1;

    ITKImage mask;
    if(this->use_mask_module)
        mask = this->mask_fetcher();

    q_image = ITKToQImageConverter::convert(this->image,
                                            mask,
                                            this->slice_index,
                                            this->do_rescale,
                                            this->do_multiply,
                                            this->use_window);

    this->ui->image_frame->setUpdatesEnabled(false);

    QPixmap pixmap = QPixmap::fromImage(*q_image);
    emit this->pixmapPainted(&pixmap);  // other modules paint into it here

    this->ui->image_frame->setPixmap(pixmap);
    this->ui->image_frame->setMinimumWidth(image.width);
    this->ui->image_frame->setMinimumHeight(image.height);

    this->ui->image_frame->setUpdatesEnabled(true);

}

void ImageViewWidget::paintEvent(QPaintEvent *paint_event)
{
    QWidget::paintEvent(paint_event);
    this->paintImage();
}

void ImageViewWidget::mouseReleaseEvent(QMouseEvent *)
{
    emit this->mouseReleasedOnImage();
}


bool ImageViewWidget::eventFilter(QObject *watched, QEvent *event)
{
    if(watched == this->ui->image_frame && event->type() == QEvent::MouseMove)
    {
        auto mouse_event = dynamic_cast<QMouseEvent*>(event);
        if(mouse_event == nullptr)
            return false;

        if(this->image.isNull())
            return false;

        QPoint position = this->ui->image_frame->mapFromGlobal(mouse_event->globalPos());

        //std::cout << "mouse move at " << position.x() << "|" << position.y() << std::endl;
        auto index = ITKImage::indexFromPoint(position,
                                              this->slice_index);
        emit this->mouseMoveOnImage(mouse_event->buttons(), index);

        return true;
    }
    return false;
}

void ImageViewWidget::wheelEvent(QWheelEvent *wheel_event)
{
    emit this->mouseWheelOnImage(wheel_event->delta());
}

void ImageViewWidget::mousePressEvent(QMouseEvent * mouse_event)
{
    if(this->image.isNull())
        return;

    QPoint position = this->ui->image_frame->mapFromGlobal(mouse_event->globalPos());
    // std::cout << "mouse pressed at " << position.x() << "|" << position.y() << std::endl;

    auto index = ITKImage::indexFromPoint(position, this->slice_index);
    if(!this->image.contains(index))
        return;

    emit this->mousePressedOnImage(mouse_event->button(), index);
}

void ImageViewWidget::repaintImage()
{
    this->paintImage(true);
}

void ImageViewWidget::repaintImageOverlays()
{
    if(this->image.isNull())
        return;

    if(q_image == nullptr)
        this->paintImage(true);

    QPixmap pixmap = QPixmap::fromImage(*q_image);
    emit this->pixmapPainted(&pixmap);  // other modules paint into it here

    this->ui->image_frame->setPixmap(pixmap);
}

void ImageViewWidget::setImage(ITKImage image)
{
    this->image = image;
    this->paintImage(true);
    emit this->imageChanged(image);
}

ITKImage ImageViewWidget::getImage() const
{
    return this->image;
}

void ImageViewWidget::save_file_with_overlays(QString file_name)
{
    if(this->ui->image_frame->pixmap() == nullptr)
        return;

    if(file_name == "")
        file_name = QFileDialog::getSaveFileName(this, "save image file with overlays");

    if(file_name.isNull())
        return;

    // 0 ... choose format form filename
    // 100 ... uncompressed
    bool saved = this->ui->image_frame->pixmap()->save(file_name, 0, 100);
    this->setStatusText( (saved ? "saved " : "error while saving ") + file_name);
}

void ImageViewWidget::load_color_to_view_only(QString file_name)
{
    if(file_name == "")
        file_name = QFileDialog::getOpenFileName(this, "open color file");

    if(file_name == QString::null || !QFile(file_name).exists())
        return;

    this->q_image = new QImage(file_name);
    this->repaintImageOverlays();
}

void ImageViewWidget::handleImageChange(ITKImage image)
{
    this->setImage(image);
}
