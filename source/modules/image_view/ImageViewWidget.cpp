#include "ImageViewWidget.h"
#include "ui_ImageViewWidget.h"

#include "ITKToQImageConverter.h"
#include <QVBoxLayout>
#include <QFileDialog>

#include "CrosshairModule.h"

ImageViewWidget::ImageViewWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::ImageViewWidget),
    inner_image_frame(nullptr),
    q_image(nullptr),
    image(ITKImage::Null),
    slice_index(0)
{
    ui->setupUi(this);

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
        if(inner_image_frame != nullptr)
            inner_image_frame->setPixmap(QPixmap());
        return;
    }

    if(repaint && q_image != nullptr) {
        delete q_image;
        q_image = nullptr;
    }

    if(q_image != nullptr)
        return;

    q_image = ITKToQImageConverter::convert(this->image,
                                            this->slice_index);

    this->ui->image_frame->setUpdatesEnabled(false);

    if(inner_image_frame != nullptr)
        delete inner_image_frame;
    inner_image_frame = new QLabel(this->ui->image_frame);
    inner_image_frame->setMouseTracking(true);
    inner_image_frame->installEventFilter(this);
    inner_image_frame->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    inner_image_frame->setMinimumSize(q_image->size());
    //  frame->setMaximumSize(q_image.size());
    //  frame->setBaseSize(q_image.size());
    //  frame->setFixedSize(q_image.size());

    QPixmap pixmap = QPixmap::fromImage(*q_image);
    emit this->pixmapPainted(&pixmap);  // other modules paint into it here

    inner_image_frame->setPixmap(pixmap);
    inner_image_frame->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    QVBoxLayout* layout = new QVBoxLayout();
    layout->addWidget(inner_image_frame);
    if(this->ui->image_frame->layout() != nullptr)
        delete this->ui->image_frame->layout();
    this->ui->image_frame->setLayout(layout);
    this->ui->image_frame->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    this->ui->image_frame->setMinimumSize(q_image->size());

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


bool ImageViewWidget::eventFilter(QObject *target, QEvent *event)
{
    if(this->image.isNull())
        return false;

    if(event->type() == QEvent::Paint)
        return false;

    if(event->type() == QEvent::MouseMove && target == inner_image_frame)
    {
        QMouseEvent* mouse_event = static_cast<QMouseEvent*>(event);
        if(mouse_event == nullptr)
            return false;

        QPoint position = this->inner_image_frame->mapFromGlobal(mouse_event->globalPos());

        //std::cout << "mouse move at " << position.x() << "|" << position.y() << std::endl;
        auto index = ITKImage::indexFromPoint(position,
                                              this->slice_index);
        emit this->mouseMoveOnImage(mouse_event->buttons(), index);
    }

    if(event->type() == QEvent::Wheel && target == inner_image_frame)
    {
        QWheelEvent* wheel_event = static_cast<QWheelEvent*>(event);
        if(wheel_event == nullptr)
            return false;

        emit this->mouseWheelOnImage(wheel_event->delta());
    }

    return false; // always returning false, so the pixmap is painted
}


void ImageViewWidget::mousePressEvent(QMouseEvent * mouse_event)
{
    if(this->image.isNull() || this->inner_image_frame == nullptr)
        return;

    QPoint position = this->inner_image_frame->mapFromGlobal(mouse_event->globalPos());
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

    inner_image_frame->setPixmap(pixmap);
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
    if(this->inner_image_frame == nullptr ||
            this->inner_image_frame->pixmap() == nullptr)
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save image file with overlays");
    if(file_name.isNull())
        return;

    // 0 ... choose format form filename
    // 100 ... uncompressed
    bool saved = this->inner_image_frame->pixmap()->save(file_name, 0, 100);
    this->setStatusText( (saved ? "saved " : "error while saving ") + file_name);
}

void ImageViewWidget::handleImageChange(ITKImage image)
{
    this->setImage(image);
}
