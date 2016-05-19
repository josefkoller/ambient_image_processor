#ifndef IMAGEWIDGET_H
#define IMAGEWIDGET_H

#include <QWidget>
#include <QList>
#include <QLabel>

#include "ITKImageProcessor.h"
#include "ITKToQImageConverter.h"

#include <functional>
#include <thread>

#include <QListWidgetItem>

namespace Ui {
class ImageWidget;
}

Q_DECLARE_METATYPE(ITKImageProcessor::ImageType::Pointer);

class ImageWidget : public QWidget
{
    Q_OBJECT
public:
    typedef ITKImageProcessor::ImageType Image;

    explicit ImageWidget(QWidget *parent = 0);
    ~ImageWidget();

    void setImage(const Image::Pointer& image);

    void saveImageState();
    void restoreImageState();
    void shrink(unsigned int shrink_factor_x,
                unsigned int shrink_factor_y,
                unsigned int shrink_factor_z);

    void showStatistics();
    void hideStatistics();
    void showSliceControl();
    void showHistogram();
    void hideHistogram();

    void showImageOnly();

    void connectSliceControlTo(ImageWidget* other_image_widget);
    void connectProfileLinesTo(ImageWidget* other_image_widget);
private slots:
    void on_slice_slider_valueChanged(int value);

    void on_histogram_bin_count_spinbox_editingFinished();

    void on_update_histogram_button_clicked();

    void on_update_window_spinbox_clicked();

    void histogram_mouse_move(QMouseEvent*);
    void info_box_toggled(bool arg1);


    void on_histogram_box_outer_toggled(bool arg1);

    void on_shrink_button_clicked();

    void on_restore_original_button_clicked();

    void on_load_button_clicked();

    void on_save_button_clicked();

    void on_extract_button_clicked();

    void on_from_x_spinbox_editingFinished();

public:
    Ui::ImageWidget *ui;
private:
    ImageWidget* output_widget;
    ImageWidget* output_widget2;
    ImageWidget* output_widget3;

    Image::Pointer image_save;

    Image::Pointer image;
    uint slice_index;

    bool show_pixel_value_at_cursor;

    bool show_statistics;
    bool show_slice_control;
    bool show_histogram;

    void setSliceIndex(uint slice_index);
    uint userSliceIndex() const;

    void paintImage();
    void statistics();
    void calculateHistogram();

    void userWindow(Image::PixelType& window_from,
                    Image::PixelType& window_to);

    void setPixelInfo(QPoint position, double pixel_value);

    int selectedReferenceROI();
    void paintSelectedReferenceROI();
    void updateReferenceROI();

    void displayOriginAndSpacing();
    void setInputRanges();

    void paintSelectedProfileLineInImage();
signals:
    void sliceIndexChanged(uint slice_index);
public slots:
    void connectedSliceControlChanged(uint slice_index);

public:
    Image::Pointer getImage() { return this->image; }
private slots:
    void updateExtractedSizeLabel(int);
    void on_restore_original_button_extract_clicked();
    void on_slice_spinbox_valueChanged(int arg1);
    void on_fromMinimumButton_clicked();
    void on_toMaximumButton_clicked();

public:
    void setMinimumSizeToImage();
    void hidePixelValueAtCursor();
    void showPixelValueAtCursor();
    void setOutputWidget(ImageWidget* output_widget);
    void setOutputWidget2(ImageWidget* output_widget);
    void setOutputWidget3(ImageWidget* output_widget);

    ImageWidget* getOutputWidget() const;

    void setPage(unsigned char page_index);
protected:
    void paintEvent(QPaintEvent *);
    void mousePressEvent(QMouseEvent *);
    void mouseReleaseEvent(QMouseEvent *);

private:
    std::thread* worker_thread;

signals:
    void fireWorkerFinished();
    void fireStatusTextChange(QString text);
    void fireImageChange(Image::Pointer image);
private slots:
    void handleWorkerFinished();
    void handleStatusTextChange(QString text);
    void handleImageChange(Image::Pointer image);

    void on_pushButton_clicked();

    void on_thresholdButton_clicked();

    void on_pushButton_4_clicked();

    void on_referenceROIsListWidget_currentRowChanged(int currentRow);

    void on_referenceROIsListWidget_itemSelectionChanged();

    void on_pushButton_6_clicked();

private:
    QLabel* inner_image_frame;
    QImage* q_image;
protected:
    bool eventFilter(QObject *target, QEvent *event);

private:
    bool adding_reference_roi;
    QList<QVector<QPoint>> reference_rois;

    std::vector<ITKImageProcessor::ReferenceROIStatistic> reference_rois_statistic;

public:
    void setReferenceROIs(QList<QVector<QPoint>> reference_rois);

public slots:
    void handleRepaintImage();
};

#endif // IMAGEWIDGET_H
