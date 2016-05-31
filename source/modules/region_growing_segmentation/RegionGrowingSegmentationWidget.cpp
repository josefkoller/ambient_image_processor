#include "RegionGrowingSegmentationWidget.h"
#include "ui_RegionGrowingSegmentationWidget.h"

#include <QMouseEvent>

#include "RegionGrowingSegmentationProcessor.h"
#include "SegmentsToLabelImageConverter.h"


#include "../non_local_gradient/NonLocalGradientProcessor.h"

#include <QFileDialog>
#include <QTextStream>

RegionGrowingSegmentationWidget::RegionGrowingSegmentationWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::RegionGrowingSegmentationWidget),
    is_adding_seed_point(false),
    kernel_sigma_fetcher(nullptr),
    kernel_size_fetcher(nullptr),
    image(ITKImage::Null)
{
    ui->setupUi(this);
}

RegionGrowingSegmentationWidget::~RegionGrowingSegmentationWidget()
{
    delete ui;
}

void RegionGrowingSegmentationWidget::on_newSegmentButton_clicked()
{
    this->addSegment();
}

void RegionGrowingSegmentationWidget::addSegment(QString name)
{
    RegionGrowingSegmentation::Segment segment = region_growing_segmentation.addSegment();

    if(name != "")
    {
        segment.name = name.toStdString();
        region_growing_segmentation.setSegmentName(
                    region_growing_segmentation.getSegments().size() - 1, name.toStdString());
    }

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
        this->setStatusText("No segment selected");
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
        this->setStatusText("aborted new seed point selection");
        return;
    }

    int segment_index = this->selectedRow(this->ui->segmentsListWidget);
    if(segment_index == -1)
    {
        this->setStatusText("No segment selected");
        return;
    }
    this->is_adding_seed_point = true;
    this->ui->newSeedButton->setFlat(true);
    this->setStatusText("Select the position of the new seed point");
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
        this->setStatusText("No segment selected");
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
    return ITKImage::indexToText(point);
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
        this->setStatusText("No seed point selected");
        return;
    }
    int segment_index = this->selectedRow(this->ui->segmentsListWidget);
    if(segment_index == -1)
    {
        this->setStatusText("No segment selected");
        return;
    }
    this->region_growing_segmentation.removeSeedPoint(segment_index, seed_index);
    delete this->ui->seedsListWidget->item(seed_index);

}

void RegionGrowingSegmentationWidget::on_performSegmentationButton_clicked()
{
    this->processInWorkerThread();
}

ITKImage RegionGrowingSegmentationWidget::processImage(ITKImage source_image)
{
    if(this->kernel_sigma_fetcher == nullptr || this->kernel_size_fetcher == nullptr)
        return ITKImage();

    // input...
    typedef SegmentsToLabelImageConverter::LabelImage LabelImage;

    float tolerance = this->ui->toleranceSpinbox->value();

    float kernel_sigma = this->kernel_sigma_fetcher();
    float kernel_size = this->kernel_size_fetcher();

    ITKImage gradient_image = NonLocalGradientProcessor::process(
                source_image, kernel_size, kernel_sigma);


    // action...
    this->label_image = RegionGrowingSegmentationProcessor::process(
                gradient_image,
                this->region_growing_segmentation.getSegments(),
                tolerance);

    label_image.getPointer()->SetOrigin(source_image.getPointer()->GetOrigin());
    label_image.getPointer()->SetSpacing(source_image.getPointer()->GetSpacing());

    return label_image;
}

void RegionGrowingSegmentationWidget::setKernelSigmaFetcher(std::function<float()> kernel_sigma_fetcher)
{
    this->kernel_sigma_fetcher = kernel_sigma_fetcher;
}

void RegionGrowingSegmentationWidget::setKernelSizeFetcher(
        std::function<uint()> kernel_size_fetcher)
{
    this->kernel_size_fetcher = kernel_size_fetcher;
}

void RegionGrowingSegmentationWidget::on_saveParameterButton_clicked()
{
    QString file_path = QFileDialog::getSaveFileName(this, "Parameter File", "", "*.txt");

    QString data = QString::number(this->ui->toleranceSpinbox->value()) + "\n";
    auto segments = this->region_growing_segmentation.getSegmentObjects();
    for(RegionGrowingSegmentation::Segment segment : segments)
    {
        data += QString::fromStdString(segment.name) + ":";
        for(RegionGrowingSegmentation::Position seed_point : segment.seed_points)
        {
            data += QString::number(seed_point[0]) + "|" +
                    QString::number(seed_point[1]) + "|" +
                    QString::number(seed_point[2]) + ";";
        }
        data += "\n";
    }

    QFile file(file_path);
    if(file.open(QIODevice::WriteOnly))
    {
        QTextStream stream(&file);
        stream << data;
    }
}



void RegionGrowingSegmentationWidget::on_load_ParameterButton_clicked()
{
    QString file_path = QFileDialog::getOpenFileName(this, "Parameter File", "", "*.txt");
    QFile file(file_path);
    if(!file.open(QIODevice::ReadOnly))
        return;
    QTextStream stream(&file);
    QString data = stream.readAll();

    QStringList elements = data.split("\n");
    QString tolerance_text = elements[0];
    this->ui->toleranceSpinbox->setValue(tolerance_text.toFloat());

    this->region_growing_segmentation.clear();
    this->ui->segmentsListWidget->clear();
    this->ui->seedsListWidget->clear();

    for(int i = 1; i < elements.length(); i++)
    {
        if(elements[i].trimmed() == "")
            continue;

        QStringList segment_elements = elements[i].split(":");
        QString segment_name = segment_elements[0];

        this->addSegment(segment_name);
        uint segment_index = i - 1;

        QStringList seed_points_elements = segment_elements[1].split(";");

        for(int s = 0; s < seed_points_elements.length(); s++)
        {
            if(seed_points_elements[s].trimmed() == "")
                continue;

            QStringList seed_point_elements = seed_points_elements[s].split("|");
            uint x = seed_point_elements[0].toInt();
            uint y = seed_point_elements[1].toInt();
            uint z = seed_point_elements[2].toInt();
            RegionGrowingSegmentation::Position seed_point;
            seed_point[0] = x;
            seed_point[1] = y;
            seed_point[2] = z;

            this->region_growing_segmentation.addSeedPoint(segment_index, seed_point);
        }
    }

    this->ui->segmentsListWidget->setCurrentRow(0);
    this->refreshSeedPointGroupBoxTitle();
    this->refreshSeedPointList();
}

std::vector<std::vector<RegionGrowingSegmentation::Position> > RegionGrowingSegmentationWidget::getSegments() const
{
    return this->region_growing_segmentation.getSegments();
}

RegionGrowingSegmentationWidget::LabelImage RegionGrowingSegmentationWidget::getLabelImage() const
{
    return this->label_image;
}

void RegionGrowingSegmentationWidget::registerModule(ImageWidget* image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    connect(image_widget, &ImageWidget::mousePressedOnImage,
            this, &RegionGrowingSegmentationWidget::mousePressedOnImage);

    connect(image_widget, &ImageWidget::imageChanged,
            this, [this] (ITKImage image) {
        this->image = image;
    });
}

void RegionGrowingSegmentationWidget::mousePressedOnImage(Qt::MouseButton button,
                                                          ITKImage::Index cursor_index)
{
    if(this->isAddingSeedPoint())
    {
        this->addSeedPointAt(cursor_index);
    }

}

