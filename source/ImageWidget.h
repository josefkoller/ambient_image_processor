#ifndef IMAGEWIDGET_H
#define IMAGEWIDGET_H

#include <QWidget>
#include <QList>
#include <QLabel>

#include "ITKImageProcessor.h"
#include "ITKToQImageConverter.h"

#include <functional>
#include <thread>

#include "retinex/MultiScaleRetinex.h"

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

    void line_profile_mouse_move(QMouseEvent*);

    void on_histogram_box_outer_toggled(bool arg1);

    void on_add_profile_line_button_clicked();

    void on_line_profile_list_widget_itemSelectionChanged();

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

    struct ProfileLine {
    private:
        QPoint _position1;
        QPoint _position2;
        bool position1_is_set;
        bool position2_is_set;
    public:
        ProfileLine() : position1_is_set(false), position2_is_set(false)
        {}
        void setPosition1(QPoint position1) {
            this->_position1 = position1;
            this->position1_is_set = true;
        }
        void setPosition2(QPoint position2) {
            this->_position2 = position2;
            this->position2_is_set = true;
        }
        QPoint position1()
        {
            return this->_position1;
        }
        QPoint position2()
        {
            return this->_position2;
        }
        bool isSet()
        {
            return this->position1_is_set && this->position2_is_set;
        }

        QString text()
        {
            if(this->isSet())
            {
                return QString("%1 | %2  -  %3 | %4").arg(
                        QString::number(_position1.x()),
                        QString::number(_position1.y()),
                        QString::number(_position2.x()),
                        QString::number(_position2.y()) );
            }
            if(this->position1_is_set)
            {
                return "only position1 set";
            }
            if(this->position2_is_set)
            {
                return "only position2 set";
            }
            return "empty line";
        }
    };
    QList<ProfileLine> profile_lines;
    bool adding_profile_line;
    void paintSelectedProfileLine();
    int selectedProfileLineIndex();
    void paintSelectedProfileLineInImage();

    void displayOriginAndSpacing();
    void setInputRanges();

    ImageWidget* profile_line_parent;
signals:
    void sliceIndexChanged(uint slice_index);
    void profileLinesChanged();
    void selectedProfileLineIndexChanged(int selected_index);
public slots:
    void connectedSliceControlChanged(uint slice_index);
    void connectedProfileLinesChanged();
    void connectedSelectedProfileLineIndexChanged(int selected_index);

public:
    Image::Pointer getImage() { return this->image; }
    QList<ImageWidget::ProfileLine> getProfileLines() { return this->profile_lines; }
    int getSelectedProfileLineIndex();
private slots:
    void updateExtractedSizeLabel(int);
    void on_restore_original_button_extract_clicked();
    void on_slice_spinbox_valueChanged(int arg1);
    void on_fromMinimumButton_clicked();
    void on_toMaximumButton_clicked();
    void on_calculate_button_clicked();

public:
    void setMinimumSizeToImage();
    void hidePixelValueAtCursor();
    void showPixelValueAtCursor();
    void setOutputWidget(ImageWidget* output_widget);
    void setOutputWidget2(ImageWidget* output_widget);
    void setOutputWidget3(ImageWidget* output_widget);
    void setPage(unsigned char page_index);
protected:
    void paintEvent(QPaintEvent *);
    void mousePressEvent(QMouseEvent *);

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

    void on_addScaleButton_clicked();

    void on_pushButton_2_clicked();

private:
    QLabel* inner_image_frame;
    QImage* q_image;
    MultiScaleRetinex multi_scale_retinex;
protected:
    bool eventFilter(QObject *target, QEvent *event);
};

#endif // IMAGEWIDGET_H
