#include "ManualDrawMask.h"
#include "ui_ManualDrawMask.h"

#include <QPainter>

ManualDrawMask::ManualDrawMask(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::ManualDrawMask),
    is_drawing_mask(false),
    polygon_fill_rule(Qt::WindingFill)
{
    ui->setupUi(this);
}

ManualDrawMask::~ManualDrawMask()
{
    delete ui;
}

void ManualDrawMask::registerModule(ImageWidget *image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(image_widget, &ImageWidget::mouseMoveOnImage,
            this, &ManualDrawMask::mouseMoveOnImage);
    connect(image_widget, &ImageWidget::mouseReleasedOnImage,
            this, &ManualDrawMask::mouseReleasedOnImage);
    connect(image_widget, &ImageWidget::pixmapPainted,
            this, &ManualDrawMask::paintPolygon);
    connect(this, &ManualDrawMask::repaintImage,
            image_widget, &ImageWidget::repaintImage);
}

QPolygon ManualDrawMask::createPolygon()
{
    QVector<QPoint> qpoints;
    for(ITKImage::Index point : this->polygon_points)
        qpoints.push_back(ITKImage::pointFromIndex(point));
    return QPolygon(qpoints);
}

void ManualDrawMask::paintPolygon(QPixmap* pixmap)
{
    if(!this->is_drawing_mask)
        return;

    QPolygon polygon = createPolygon();

    QPainter painter(pixmap);
    QPen pen(Qt::black);
    pen.setWidth(1);
    painter.setPen(pen);
    QColor color(0,250,0,100);
    QBrush brush(color);
    painter.setBrush(brush);
    painter.drawPolygon(polygon, this->polygon_fill_rule);
}

void ManualDrawMask::mouseReleasedOnImage()
{
    if(this->is_drawing_mask)
    {
        this->is_drawing_mask = false;
        this->ui->startButton->setFlat(false);

        this->repaintImage();
        this->processInWorkerThread();
    }
}

void ManualDrawMask::mouseMoveOnImage(Qt::MouseButtons buttons, ITKImage::Index cursor_index)
{
    bool is_left_button = (buttons & Qt::LeftButton) == Qt::LeftButton;

    if(this->is_drawing_mask && is_left_button)
    {
        this->polygon_points.push_back(cursor_index);

        this->repaintImage();
    }
}

ITKImage ManualDrawMask::processImage(ITKImage image)
{
    if(this->polygon_points.size() == 0)
      image.cloneSameSizeWithZeros();

    // get the slice index from the first point
    uint slice_index = this->polygon_points[0][2];

    QPolygon polygon = createPolygon();
    ITKImage mask = image.clone();
    mask.setEachPixel([&polygon, this, slice_index]
        (uint x, uint y, uint z) {
        if(z != slice_index)
          return false;

        QPoint point(x, y);
        return polygon.containsPoint(point, this->polygon_fill_rule);
    });
    return mask;
}

void ManualDrawMask::on_startButton_clicked()
{
    // start drawing the polygon
    this->polygon_points.clear();
    this->is_drawing_mask = true;
    this->ui->startButton->setFlat(true);
}
