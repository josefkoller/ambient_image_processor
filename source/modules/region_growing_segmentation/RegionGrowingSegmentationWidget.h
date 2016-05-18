#ifndef REGIONGROWINGSEGMENTATIONWIDGET_H
#define REGIONGROWINGSEGMENTATIONWIDGET_H

#include <QWidget>
#include <QListWidgetItem>

#include "RegionGrowingSegmentation.h"
#include "ImageWidget.h"
#include "RegionGrowingSegmentationProcessor.h"

#include <functional>

#include "ITKImage.h"

#include "BaseModuleWidget.h"

namespace Ui {
class RegionGrowingSegmentationWidget;
}

class RegionGrowingSegmentationWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit RegionGrowingSegmentationWidget(QWidget *parent = 0);
    ~RegionGrowingSegmentationWidget();

    bool isAddingSeedPoint() const;
    void addSeedPointAt(RegionGrowingSegmentation::Position position);
private:
    Ui::RegionGrowingSegmentationWidget *ui;

    RegionGrowingSegmentation region_growing_segmentation;
    bool is_adding_seed_point;

    int selectedRow(QListWidget* list_widget) const;
    QString text(RegionGrowingSegmentation::Position point) const;
    void refreshSeedPointList();
    void refreshSeedPointGroupBoxTitle();
    void stopAddingSeedPoint();


    void addSegment(QString name = "");
private slots:
    void on_newSegmentButton_clicked();
    void on_removeSegmentButton_clicked();
    void on_newSeedButton_clicked();
    void on_segmentsListWidget_itemSelectionChanged();

    void on_segmentsListWidget_itemChanged(QListWidgetItem *item);

    void on_seedsListWidget_itemSelectionChanged();

    void on_removeSeedButton_clicked();

    void on_performSegmentationButton_clicked();

    void on_saveParameterButton_clicked();

    void on_load_ParameterButton_clicked();

signals:
    void statusTextChange(QString);

private:
    std::function<float()> kernel_sigma_fetcher;
    std::function<uint()> kernel_size_fetcher;

    RegionGrowingSegmentationProcessor::LabelImage::Pointer label_image;
public:
    void setKernelSigmaFetcher(std::function<float()> kernel_sigma_fetcher);
    void setKernelSizeFetcher(std::function<uint()> kernel_size_fetcher);

    RegionGrowingSegmentationProcessor::LabelImage::Pointer getLabelImage() const;
    std::vector<std::vector<RegionGrowingSegmentation::Position> > getSegments() const;

protected:
    virtual ITKImage processImage(ITKImage image);
};

#endif // REGIONGROWINGSEGMENTATIONWIDGET_H
