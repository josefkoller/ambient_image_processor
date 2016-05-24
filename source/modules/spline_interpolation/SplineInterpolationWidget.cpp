#include "SplineInterpolationWidget.h"
#include "ui_SplineInterpolationWidget.h"

#include <QPainter>
#include <QPen>

SplineInterpolationWidget::SplineInterpolationWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::SplineInterpolationWidget),
    adding_reference_roi(false),
    image(ITKImage::Null)
{
    ui->setupUi(this);
}

SplineInterpolationWidget::~SplineInterpolationWidget()
{
    delete ui;
}

ITKImage SplineInterpolationWidget::processImage(ITKImage image)
{
    if(this->reference_rois_statistic.size() == 0)
    {
        this->setStatusText("add some rois first");
        return ITKImage();
    }

    uint spline_order = this->ui->splineOrderSpinbox->value();
    uint spline_levels = this->ui->splineLevelsSpinbox->value();
    uint spline_control_points = this->ui->splineControlPoints->value();

    if(spline_control_points <= spline_order)
    {
        this->setStatusText("need more control points than the spline order is");
        return ITKImage();
    }

    ITKImage field_image;
    ITKImage output_image = SplineInterpolationProcessor::process(
                image, spline_order, spline_levels, spline_control_points,
                this->reference_rois_statistic, field_image);

    SplineInterpolationProcessor::printMetric(this->reference_rois_statistic);
//    target_widget->setReferenceROIs(this->reference_rois);

    return field_image;
}

void SplineInterpolationWidget::setReferenceROIs(QList<QVector<Point>> reference_rois)
{
    this->ui->referenceROIsListWidget->clear();
    this->reference_rois_statistic.clear();

    this->reference_rois = reference_rois;

    for(int i = 0; i < reference_rois.size(); i++)
    {
        QString item = "ROI" + QString::number(i);
        this->ui->referenceROIsListWidget->addItem(item);
        this->ui->referenceROIsListWidget->setCurrentRow(i);
        this->reference_rois_statistic.push_back(SplineInterpolationProcessor::ReferenceROIStatistic());

        updateReferenceROI();
    }

    SplineInterpolationProcessor::printMetric(this->reference_rois_statistic);
}

void SplineInterpolationWidget::on_pushButton_6_clicked()
{
    this->processInWorkerThread();
}

void SplineInterpolationWidget::on_add_reference_roi_button_clicked()
{
    this->adding_reference_roi = true;

    QVector<Point> roi;
    reference_rois.push_back(roi);

    uint index = reference_rois.size() - 1;

    QString item = "ROI" + QString::number(index);
    this->ui->referenceROIsListWidget->addItem(item);
    this->ui->referenceROIsListWidget->setCurrentRow(index);

    this->reference_rois_statistic.push_back(SplineInterpolationProcessor::ReferenceROIStatistic());
}

int SplineInterpolationWidget::selectedReferenceROI()
{
    if(this->ui->referenceROIsListWidget->selectionModel()->selectedIndexes().size() == 0)
    {
        return -1;
    }
    return this->ui->referenceROIsListWidget->selectionModel()->selectedIndexes().at(0).row();
}


void SplineInterpolationWidget::paintSelectedReferenceROI(QPixmap* pixmap)
{
    int index = this->selectedReferenceROI();
    if(index == -1)
    {
        return;
    }
    QVector<Point> roi = this->reference_rois[index];

    if(roi.size() == 0)
        return;

    QVector<QPoint> qpoint_roi;
    for(Point point : roi)
        qpoint_roi.push_back(ITKImage::pointFromIndex(point));

    QPolygon polygon(qpoint_roi);

    QPainter painter(pixmap);

    QPen pen(Qt::black);
    pen.setWidth(1);
    painter.setPen(pen);

    QColor color(0,250,0,100);
    QBrush brush(color);
    painter.setBrush(brush);

    painter.drawPolygon(polygon);
}

void SplineInterpolationWidget::mouseReleasedOnImage()
{
    if(this->adding_reference_roi)
    {
        this->adding_reference_roi = false;
        updateReferenceROI();
        emit this->repaintImage();
    }
}

void SplineInterpolationWidget::mouseMoveOnImage(Qt::MouseButtons buttons, ITKImage::Index cursor_index)
{
    bool is_left_button = (buttons & Qt::LeftButton) == Qt::LeftButton;

    int selected_roi_index = this->selectedReferenceROI();

    if(this->adding_reference_roi &&
            is_left_button &&
            selected_roi_index > -1)
    {
        std::cout << "add point" << std::endl;

        this->reference_rois[selected_roi_index].push_back(cursor_index);
    }
}
void SplineInterpolationWidget::updateReferenceROI()
{
    int index = this->selectedReferenceROI();
    if(index == -1)
        return;

    if(this->image.isNull())
        return;

    QVector<Point> roi = this->reference_rois[index];

    SplineInterpolationProcessor::ReferenceROIStatistic statistic =
            SplineInterpolationProcessor::calculateStatisticInROI(roi, this->image);

    QString text = QString("Position: %1, Mean: %3").arg(
                ITKImage::indexToText(statistic.point),
                QString::number(statistic.mean_value));

    this->ui->referenceROIsListWidget->item(index)->setText(text);

    this->reference_rois_statistic[index] = statistic;

  //  emit this->repaintImage();
}


void SplineInterpolationWidget::registerModule(ImageWidget* image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(image_widget, &ImageWidget::pixmapPainted,
            this, &SplineInterpolationWidget::paintSelectedReferenceROI);
    connect(image_widget, &ImageWidget::mouseMoveOnImage,
            this, &SplineInterpolationWidget::mouseMoveOnImage);
    connect(image_widget, &ImageWidget::mouseReleasedOnImage,
            this, &SplineInterpolationWidget::mouseReleasedOnImage);

    connect(image_widget, &ImageWidget::imageChanged,
            this, [this] (ITKImage itk_image) {
        this->image = itk_image;
    });

    connect(this, &SplineInterpolationWidget::repaintImage,
            image_widget, &ImageWidget::repaintImage);

}

void SplineInterpolationWidget::on_referenceROIsListWidget_currentRowChanged(int currentRow)
{
}

void SplineInterpolationWidget::on_referenceROIsListWidget_itemSelectionChanged()
{
    emit this->repaintImage();
}
