#include "ImageViewWidget.h"
#include "ui_ImageViewWidget.h"

#include "ITKToQImageConverter.h"
#include <QVBoxLayout>
#include <QFileDialog>

#include "CrosshairModule.h"

ImageViewWidget::ImageViewWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::ImageViewWidget),
    q_image(nullptr),
    image(ITKImage::Null),
    slice_index(0)
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

    q_image = ITKToQImageConverter::convert(this->image,
                                            this->slice_index);

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

void ImageViewWidget::save_file_with_overlays()
{
    if(this->ui->image_frame->pixmap() == nullptr)
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save image file with overlays");
    if(file_name.isNull())
        return;

    // 0 ... choose format form filename
    // 100 ... uncompressed
    bool saved = this->ui->image_frame->pixmap()->save(file_name, 0, 100);
    this->setStatusText( (saved ? "saved " : "error while saving ") + file_name);
}

void ImageViewWidget::load_color_to_view_only_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "open color file");
    if(file_name == QString::null || !QFile(file_name).exists())
        return;

    this->q_image = new QImage(file_name);
    this->repaintImageOverlays();
}

void ImageViewWidget::handleImageChange(ITKImage image)
{
    this->setImage(image);
}
