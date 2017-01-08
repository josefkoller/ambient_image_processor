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
    typedef RegionGrowingSegmentationProcessor::LabelImage LabelImage;

    explicit RegionGrowingSegmentationWidget(QString title, QWidget *parent = 0);
    ~RegionGrowingSegmentationWidget();

    bool isAddingSeedPoint() const;
    void addSeedPointAt(RegionGrowingSegmentation::Position position);
private:
    Ui::RegionGrowingSegmentationWidget *ui;

    ITKImage image;

    RegionGrowingSegmentation region_growing_segmentation;
    bool is_adding_seed_point;

    LabelImage label_image;

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

public:
    std::vector<std::vector<RegionGrowingSegmentation::Position> > getSegments() const;
    LabelImage getLabelImage() const;

protected:
    virtual ITKImage processImage(ITKImage image);

public:
    void registerModule(ImageWidget* image_widget);

private slots:
    void mousePressedOnImage(Qt::MouseButton button, ITKImage::Index cursor_index);
    void on_toleranceSpinbox_valueChanged(double arg1);
};

#endif // REGIONGROWINGSEGMENTATIONWIDGET_H
