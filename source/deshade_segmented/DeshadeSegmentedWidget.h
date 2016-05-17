#ifndef DESHADESEGMENTEDWIDGET_H
#define DESHADESEGMENTEDWIDGET_H

#include <QWidget>

#include <functional>
#include <itkImage.h>

namespace Ui {
class DeshadeSegmentedWidget;
}

class DeshadeSegmentedWidget : public QWidget
{
    Q_OBJECT

public:
    explicit DeshadeSegmentedWidget(QWidget *parent = 0);
    ~DeshadeSegmentedWidget();

    typedef itk::Image<double> Image;
    typedef Image::IndexType SeedPoint;
    typedef std::vector<SeedPoint> Segment;
    typedef std::vector<Segment> Segments;
    typedef std::function<Segments()> SegmentsFetcher;
    typedef itk::Image<unsigned char> LabelImage;
    typedef std::function<LabelImage::Pointer()> LabelImageFetcher;
    typedef std::function<Image::Pointer()> SourceImageFetcher;
    typedef std::function<void(Image::Pointer, Image::Pointer)> ResultProcessor;
private slots:
    void on_perform_button_clicked();

private:
    Ui::DeshadeSegmentedWidget *ui;
    SegmentsFetcher segment_fetcher;
    LabelImageFetcher label_image_fetcher;
    SourceImageFetcher source_image_fetcher;
    ResultProcessor result_processor;

public:
    void setSegmentsFetcher(SegmentsFetcher segment_fetcher);
    void setLabelImageFetcher(LabelImageFetcher label_image_fetcher);
    void setSourceImageFetcher(SourceImageFetcher source_image_fetcher);
    void setResultProcessor(ResultProcessor result_processor);
};

#endif // DESHADESEGMENTEDWIDGET_H
