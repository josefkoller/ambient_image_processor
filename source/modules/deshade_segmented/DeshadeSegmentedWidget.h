#ifndef DESHADESEGMENTEDWIDGET_H
#define DESHADESEGMENTEDWIDGET_H

#include <QWidget>

#include <functional>

#include "ITKImage.h"
#include "BaseModuleWidget.h"

namespace Ui {
class DeshadeSegmentedWidget;
}

class DeshadeSegmentedWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit DeshadeSegmentedWidget(ImageWidget *parent = 0);
    ~DeshadeSegmentedWidget();

    typedef ITKImage::InnerITKImage Image;
    typedef Image::IndexType SeedPoint;
    typedef std::vector<SeedPoint> Segment;
    typedef std::vector<Segment> Segments;
    typedef std::function<Segments()> SegmentsFetcher;
    typedef itk::Image<unsigned char> LabelImage;
    typedef std::function<LabelImage::Pointer()> LabelImageFetcher;
private slots:
    void on_perform_button_clicked();

private:
    Ui::DeshadeSegmentedWidget *ui;
    SegmentsFetcher segment_fetcher;
    LabelImageFetcher label_image_fetcher;

public:
    void setSegmentsFetcher(SegmentsFetcher segment_fetcher);
    void setLabelImageFetcher(LabelImageFetcher label_image_fetcher);

protected:
    virtual ITKImage processImage(ITKImage image);
};

#endif // DESHADESEGMENTEDWIDGET_H
