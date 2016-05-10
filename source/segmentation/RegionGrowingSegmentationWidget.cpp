#include "RegionGrowingSegmentationWidget.h"
#include "ui_RegionGrowingSegmentationWidget.h"

#include <QMouseEvent>

#include "RegionGrowingSegmentationProcessor.h"
#include "SegmentsToLabelImageConverter.h"

#include <itkCastImageFilter.h>

RegionGrowingSegmentationWidget::RegionGrowingSegmentationWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::RegionGrowingSegmentationWidget),
    is_adding_seed_point(false),
    source_image_widget(nullptr),
    target_image_widget(nullptr)
{
    ui->setupUi(this);
}

RegionGrowingSegmentationWidget::~RegionGrowingSegmentationWidget()
{
    delete ui;
}

void RegionGrowingSegmentationWidget::on_newSegmentButton_clicked()
{
    RegionGrowingSegmentation::Segment segment = region_growing_segmentation.addSegment();
    this->ui->segmentsListWidget->addItem(QString::fromStdString(segment.name));
    QListWidgetItem* item = this->ui->segmentsListWidget->item(this->ui->segmentsListWidget->count()-1);
    item->setFlags(item->flags() | Qt::ItemIsEditable);
    item->setSelected(true);
}

void RegionGrowingSegmentationWidget::on_removeSegmentButton_clicked()
{
    int segment_index = this->selectedRow(this->ui->segmentsListWidget);
    if(segment_index == -1)
    {
        emit statusTextChange("No segment selected");
        return;
    }
    this->region_growing_segmentation.removeSegment(segment_index);
    delete this->ui->segmentsListWidget->item(segment_index);
}

int RegionGrowingSegmentationWidget::selectedRow(QListWidget* list_widget) const
{
    if(list_widget->selectionModel()->selectedIndexes().size() == 0)
        return -1;
    return list_widget->selectionModel()->selectedIndexes().at(0).row();

}

void RegionGrowingSegmentationWidget::on_newSeedButton_clicked()
{
    if(this->is_adding_seed_point)
    {
        // abort
        this->is_adding_seed_point = false;
        this->ui->newSeedButton->setFlat(false);
        emit statusTextChange("aborted new seed point selection");
        return;
    }

    int segment_index = this->selectedRow(this->ui->segmentsListWidget);
    if(segment_index == -1)
    {
        emit statusTextChange("No segment selected");
        return;
    }
    this->is_adding_seed_point = true;
    this->ui->newSeedButton->setFlat(true);
    emit statusTextChange("Select the position of the new seed point");
}

bool RegionGrowingSegmentationWidget::isAddingSeedPoint() const
{
    return this->is_adding_seed_point;
}

void RegionGrowingSegmentationWidget::addSeedPointAt(RegionGrowingSegmentation::Position position)
{
    int segment_index = this->selectedRow(this->ui->segmentsListWidget);
    if(segment_index == -1)
    {
        emit statusTextChange("No segment selected");
        return;
    }
    this->region_growing_segmentation.addSeedPoint(segment_index, position);
    this->refreshSeedPointList();

    //this->stopAddingSeedPoint();
}

void RegionGrowingSegmentationWidget::stopAddingSeedPoint()
{
    this->is_adding_seed_point = false;
    this->ui->newSeedButton->setFlat(false);
}

void RegionGrowingSegmentationWidget::refreshSeedPointList()
{
    this->ui->seedsListWidget->clear();
    int segment_index = this->selectedRow(this->ui->segmentsListWidget);
    if(segment_index == -1)
        return;
    std::vector<RegionGrowingSegmentation::Position> seed_points =
            this->region_growing_segmentation.getSeedPointsOfSegment(segment_index);
    for(uint row = 0; row < seed_points.size(); row++)
    {
        QString text = this->text(seed_points[row]);
        this->ui->seedsListWidget->addItem(text);
    }
}

QString RegionGrowingSegmentationWidget::text(RegionGrowingSegmentation::Position point) const
{
    return QString::number(point[0]) + "|" + QString::number(point[1]);
}

void RegionGrowingSegmentationWidget::on_segmentsListWidget_itemSelectionChanged()
{
    this->refreshSeedPointList();
    this->refreshSeedPointGroupBoxTitle();

    this->stopAddingSeedPoint();

    // remove button enabled state...
    int segment_index = this->selectedRow(this->ui->segmentsListWidget);
    this->ui->removeSegmentButton->setEnabled(segment_index > -1);
}

void RegionGrowingSegmentationWidget::refreshSeedPointGroupBoxTitle()
{
    int segment_index = this->selectedRow(this->ui->segmentsListWidget);
    if(segment_index == -1)
    {
        this->ui->seed_point_groupbox->setTitle("Seed Points");
        return;
    }
    QString name = this->ui->segmentsListWidget->item(segment_index)->text();
    this->ui->seed_point_groupbox->setTitle("Seed Points of " + name);
}

void RegionGrowingSegmentationWidget::on_segmentsListWidget_itemChanged(QListWidgetItem *item)
{
    refreshSeedPointGroupBoxTitle();

    // change the segment name...
    int segment_index = this->ui->segmentsListWidget->row(item);
    if(segment_index == -1)
        return;
    QString name = this->ui->segmentsListWidget->item(segment_index)->text();
    this->region_growing_segmentation.setSegmentName(segment_index, name.toStdString());
}

void RegionGrowingSegmentationWidget::on_seedsListWidget_itemSelectionChanged()
{
    // remove button enabled state...
    int point_index = this->selectedRow(this->ui->seedsListWidget);
    this->ui->removeSeedButton->setEnabled(point_index > -1);
}

void RegionGrowingSegmentationWidget::on_removeSeedButton_clicked()
{
    int seed_index = this->selectedRow(this->ui->seedsListWidget);
    if(seed_index == -1)
    {
        emit statusTextChange("No seed point selected");
        return;
    }
    int segment_index = this->selectedRow(this->ui->segmentsListWidget);
    if(segment_index == -1)
    {
        emit statusTextChange("No segment selected");
        return;
    }
    this->region_growing_segmentation.removeSeedPoint(segment_index, seed_index);
    delete this->ui->seedsListWidget->item(seed_index);

}

void RegionGrowingSegmentationWidget::on_performSegmentationButton_clicked()
{
    // input...
    typedef SegmentsToLabelImageConverter::LabelImage LabelImage;
    typedef ITKImageProcessor::ImageType SourceImage;
    SourceImage::Pointer source_image = this->source_image_widget->getImage();

    float tolerance = this->ui->toleranceSpinbox->value();

    // action...
    LabelImage::Pointer output_labels = RegionGrowingSegmentationProcessor::process(
                source_image, this->region_growing_segmentation.getSegments(), tolerance);

    // output...
    typedef itk::CastImageFilter<LabelImage, SourceImage> CastFilter;
    CastFilter::Pointer cast_filter = CastFilter::New();
    cast_filter->SetInput(output_labels);
    cast_filter->Update();
    this->target_image_widget->setImage(cast_filter->GetOutput());


    /*
    QuickView viewer;
    viewer.AddImage(image.GetPointer());
    viewer.AddImage(rescaleFilter->GetOutput());
    viewer.Visualize();
    */
}

void RegionGrowingSegmentationWidget::setSourceImageWidget(ImageWidget* source_image_widget)
{
    this->source_image_widget = source_image_widget;
}

void RegionGrowingSegmentationWidget::setTargetImageWidget(ImageWidget* target_image_widget)
{
    this->target_image_widget = target_image_widget;
}
