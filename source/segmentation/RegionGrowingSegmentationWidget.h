#ifndef REGIONGROWINGSEGMENTATIONWIDGET_H
#define REGIONGROWINGSEGMENTATIONWIDGET_H

#include <QWidget>
#include <QListWidgetItem>

#include "segmentation/RegionGrowingSegmentation.h"
#include "ImageWidget.h"

namespace Ui {
class RegionGrowingSegmentationWidget;
}

class RegionGrowingSegmentationWidget : public QWidget
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
private slots:
    void on_newSegmentButton_clicked();
    void on_removeSegmentButton_clicked();
    void on_newSeedButton_clicked();
    void on_segmentsListWidget_itemSelectionChanged();

    void on_segmentsListWidget_itemChanged(QListWidgetItem *item);

    void on_seedsListWidget_itemSelectionChanged();

    void on_removeSeedButton_clicked();

    void on_performSegmentationButton_clicked();

signals:
    void statusTextChange(QString);

private:
    ImageWidget* source_image_widget;
    ImageWidget* target_image_widget;
public:
    void setSourceImageWidget(ImageWidget* source_image_widget);
    void setTargetImageWidget(ImageWidget* target_image_widget);
};

#endif // REGIONGROWINGSEGMENTATIONWIDGET_H
