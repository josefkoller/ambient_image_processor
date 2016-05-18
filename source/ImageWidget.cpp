#include "ImageWidget.h"
#include "ui_ImageWidget.h"

#include <itkStatisticsImageFilter.h>
#include <itkShrinkImageFilter.h>

#include <QPainter>
#include <QFileDialog>
#include <QDateTime>

#include "cuda/CudaImageProcessor.h"

ImageWidget::ImageWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ImageWidget),
    image(nullptr),
    slice_index(0),
    show_statistics(true),
    show_slice_control(false),
    show_histogram(true),
    adding_profile_line(false),
    profile_line_parent(nullptr),
    output_widget(this),
    show_pixel_value_at_cursor(true),
    worker_thread(nullptr),
    inner_image_frame(nullptr),
    q_image(nullptr),
    image_save(nullptr),
    output_widget2(this),
    adding_reference_roi(false)
{
    ui->setupUi(this);

    this->ui->statistic_box->setVisible(this->show_statistics);
    this->ui->slice_control->setVisible(this->show_slice_control);
    this->ui->histogram_box->setVisible(this->show_histogram);

    this->ui->histogram_widget->setMouseTracking(true);
    connect(this->ui->histogram_widget, &QCustomPlot::mouseMove,
            this, &ImageWidget::histogram_mouse_move);

    this->ui->line_profile_widget->setMouseTracking(true);
    connect(this->ui->line_profile_widget, &QCustomPlot::mouseMove,
            this, &ImageWidget::line_profile_mouse_move);

    connect(this->ui->from_x_spinbox, SIGNAL(valueChanged(int)),
            this, SLOT(updateExtractedSizeLabel(int)));
    connect(this->ui->to_x_spinbox, SIGNAL(valueChanged(int)),
            this, SLOT(updateExtractedSizeLabel(int)));

    connect(this->ui->from_y_spinbox, SIGNAL(valueChanged(int)),
            this, SLOT(updateExtractedSizeLabel(int)));
    connect(this->ui->to_y_spinbox, SIGNAL(valueChanged(int)),
            this, SLOT(updateExtractedSizeLabel(int)));

    connect(this->ui->from_z_spinbox, SIGNAL(valueChanged(int)),
            this, SLOT(updateExtractedSizeLabel(int)));
    connect(this->ui->to_z_spinbox, SIGNAL(valueChanged(int)),
            this, SLOT(updateExtractedSizeLabel(int)));

    connect(this, SIGNAL(fireWorkerFinished()),
            this, SLOT(handleWorkerFinished()));
    connect(this, SIGNAL(fireStatusTextChange(QString)),
            this, SLOT(handleStatusTextChange(QString)));

    qRegisterMetaType<Image::Pointer>("Image::Pointer");
    connect(this, SIGNAL(fireImageChange(Image::Pointer)),
            this, SLOT(handleImageChange(Image::Pointer)));

    connect(this->ui->region_growing_segmentation_widget,
            SIGNAL(statusTextChange(QString)),
            SLOT(handleStatusTextChange(QString)));

    this->ui->region_growing_segmentation_widget->setSourceImageWidget(this);
    this->ui->region_growing_segmentation_widget->setTargetImageWidget(this);
    this->ui->region_growing_segmentation_widget->setKernelSigmaFetcher([this]() {
        return this->ui->non_local_gradient_widget->getKernelSigma();
    });
    this->ui->region_growing_segmentation_widget->setKernelSizeFetcher([this]() {
        return this->ui->non_local_gradient_widget->getKernelSize();
    });


    this->ui->non_local_gradient_widget->setSourceImageFetcher([this](){
        return ITKImageProcessor::cloneImage(this->image);
    });
    this->ui->non_local_gradient_widget->setResultProcessor( [this] (Image::Pointer image) {
        this->output_widget->setImage(image);
    });


    this->ui->deshade_segmented_widget->setSourceImageFetcher([this]() {
        return ITKImageProcessor::cloneImage(this->image);
    });
    this->ui->deshade_segmented_widget->setSegmentsFetcher([this]() {
        return this->ui->region_growing_segmentation_widget->getSegments();
    });
    this->ui->deshade_segmented_widget->setLabelImageFetcher([this]() {
        return this->ui->region_growing_segmentation_widget->getLabelImage();
    });
    this->ui->deshade_segmented_widget->setResultProcessor([this](
      Image::Pointer shading, Image::Pointer reflectance) {
        this->output_widget->setImage(shading);
        this->output_widget2->setImage(reflectance);
    });

    //this->ui->tgv_widget->registerModule(this);
    this->ui->tgv_widget->setSourceImageFetcher([this]() {
        return ITKImageProcessor::cloneImage(this->image);
    });
    this->ui->tgv_widget->setResultProcessor([this](Image::Pointer u) {
        emit this->output_widget->fireImageChange(u);
    });
    this->ui->tgv_widget->setIterationFinishedCallback([this](uint index, uint count, Image::Pointer u) {
        emit this->fireStatusTextChange(QString("iteration %1 / %2").arg(
                                     QString::number(index+1),
                                     QString::number(count)));
        emit this->output_widget->fireImageChange(u);
    });
}

ImageWidget::~ImageWidget()
{
    delete ui;
}


void ImageWidget::setImage(const Image::Pointer& image)
{
    this->q_image = nullptr; // redo converting
    this->image = ITKImageProcessor::cloneImage(image);

    this->setInputRanges();
    this->displayOriginAndSpacing();
    this->calculateHistogram();
    this->statistics();
    this->setSliceIndex(0);
    this->setMinimumSizeToImage();
    this->paintSelectedProfileLine();
}

uint ImageWidget::userSliceIndex() const
{
    return this->ui->slice_slider->value();
}

void ImageWidget::userWindow(Image::PixelType& window_from,
                             Image::PixelType& window_to)
{
    window_from = this->ui->window_from_spinbox->value();
    window_to = this->ui->window_to_spinbox->value();
}

void ImageWidget::setSliceIndex(uint slice_index)
{
    if(this->image.IsNull())
        return;

    if(slice_index < 0 ||
            slice_index >= this->image->GetLargestPossibleRegion().GetSize()[2])
    {
        std::cerr << "invalid slice_index for this image" << std::endl << std::flush;
        return;
    }


    this->slice_index = slice_index;
    if(this->ui->slice_slider->value() != this->slice_index)
    {
        this->ui->slice_slider->setValue(this->slice_index);
    }
    if(this->ui->slice_spinbox->value() != this->slice_index)
    {
        this->ui->slice_spinbox->setValue(this->slice_index);
    }
    this->paintImage();

    emit this->sliceIndexChanged(slice_index);
}

void ImageWidget::paintImage()
{
    if(this->image.IsNull())
        return;
    Image::PixelType window_from = 0;
    Image::PixelType window_to = 0;
    this->userWindow(window_from, window_to);

    if(q_image == nullptr)
    {
        q_image = ITKToQImageConverter::convert(this->image,
                                                       this->slice_index,
                                                       window_from,
                                                       window_to);
        if(inner_image_frame != nullptr)
            delete inner_image_frame;
        inner_image_frame = new QLabel(this->ui->image_frame);
        inner_image_frame->setMouseTracking(true);
        inner_image_frame->installEventFilter(this);
        inner_image_frame->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        inner_image_frame->setMinimumSize(q_image->size());
      //  frame->setMaximumSize(q_image.size());
      //  frame->setBaseSize(q_image.size());
      //  frame->setFixedSize(q_image.size());
        inner_image_frame->setPixmap(QPixmap::fromImage(*q_image));
        inner_image_frame->setAlignment(Qt::AlignTop | Qt::AlignLeft);

        QVBoxLayout* layout = new QVBoxLayout();
        layout->addWidget(inner_image_frame);
        if(this->ui->image_frame->layout() != nullptr)
            delete this->ui->image_frame->layout();
        this->ui->image_frame->setLayout(layout);
        this->ui->image_frame->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        this->ui->image_frame->setMinimumSize(q_image->size());
    }

    paintSelectedProfileLineInImage();
    paintSelectedReferenceROI();
}

void ImageWidget::statistics()
{
    if(this->image.IsNull())
        return;

    Image::RegionType region = this->image->GetLargestPossibleRegion();
    Image::SizeType size = region.GetSize();

    QString dimensions_info = "";
    int voxel_count = 1;
    for(int dimension = 0; dimension < size.GetSizeDimension(); dimension++)
    {
        dimensions_info += QString::number(size[dimension]);
        voxel_count *= size[dimension];
        if(dimension < size.GetSizeDimension() - 1)
        {
            dimensions_info += "x";
        }
    }
    this->ui->dimensions_label->setText(dimensions_info);
    this->ui->voxel_count_label->setText(QString::number(voxel_count));

    typedef itk::StatisticsImageFilter<Image> StatisticsCalculator;
    StatisticsCalculator::Pointer statistics_calculator = StatisticsCalculator::New();
    statistics_calculator->SetInput(this->image);

    int number_of_histogram_bins = ceil(sqrt(voxel_count));

    statistics_calculator->Update();

    this->ui->mean_label->setText(QString::number(
      statistics_calculator->GetMean() ));
    this->ui->standard_deviation_label->setText(QString::number(
      statistics_calculator->GetSigma() ));
    this->ui->variance_label->setText(QString::number(
      statistics_calculator->GetVariance() ));

    this->ui->standard_error_label->setText(QString::number(
      statistics_calculator->GetVariance() /
      statistics_calculator->GetMean() ));

    this->ui->minimum_label->setText(QString::number(
      statistics_calculator->GetMinimum()));
    this->ui->maximum_label->setText(QString::number(
      statistics_calculator->GetMaximum()));

    this->ui->window_from_spinbox->setMinimum(statistics_calculator->GetMinimum());
    this->ui->window_from_spinbox->setMaximum(statistics_calculator->GetMaximum());
    this->ui->window_from_spinbox->setValue(statistics_calculator->GetMinimum());

    this->ui->window_to_spinbox->setMinimum(statistics_calculator->GetMinimum());
    this->ui->window_to_spinbox->setMaximum(statistics_calculator->GetMaximum());
    this->ui->window_to_spinbox->setValue(statistics_calculator->GetMaximum());

    this->ui->fromMinimumButton->setText("From Minimum: " + QString::number(
                                             statistics_calculator->GetMinimum() ));
    this->ui->toMaximumButton->setText("To Maximum: " + QString::number(
                                             statistics_calculator->GetMaximum() ));
}

void ImageWidget::on_slice_slider_valueChanged(int user_slice_index)
{
    this->setSliceIndex(user_slice_index);
}

void ImageWidget::showStatistics()
{
    this->show_statistics = true;
    this->ui->statistic_box->setVisible(true);
    this->statistics();
}

void ImageWidget::hideStatistics()
{
    this->show_statistics = false;
    this->ui->statistic_box->setVisible(false);
}

void ImageWidget::showSliceControl()
{
    this->show_slice_control = true;
    this->ui->slice_control->setVisible(true);
}

void ImageWidget::showHistogram()
{
    this->show_histogram = true;
    this->calculateHistogram();
    this->ui->histogram_box->setVisible(true);
}
void ImageWidget::hideHistogram()
{
    this->show_histogram = false;
    this->ui->histogram_box->setVisible(false);
}

void ImageWidget::connectSliceControlTo(ImageWidget* other_image_widget)
{
    connect(other_image_widget, &ImageWidget::sliceIndexChanged,
            this, &ImageWidget::connectedSliceControlChanged);
}

void ImageWidget::connectProfileLinesTo(ImageWidget* other_image_widget)
{
    this->profile_line_parent = other_image_widget;
    connect(other_image_widget, &ImageWidget::profileLinesChanged,
            this, &ImageWidget::connectedProfileLinesChanged);
    connect(other_image_widget, &ImageWidget::selectedProfileLineIndexChanged,
            this, &ImageWidget::connectedSelectedProfileLineIndexChanged);
}


void ImageWidget::connectedSliceControlChanged(uint slice_index)
{
    this->setSliceIndex(slice_index);
}

void ImageWidget::calculateHistogram()
{
    if(this->image.IsNull())
        return;

    int bin_count = this->ui->histogram_bin_count_spinbox->value();

    Image::PixelType window_from = 0;
    Image::PixelType window_to = 0;
    this->userWindow(window_from, window_to);

    std::vector<double> intensities;
    std::vector<double> probabilities;
    ITKImageProcessor::histogram_data(this->image,
                                      bin_count,
                                      window_from, window_to,
                                      intensities, probabilities);

    this->ui->histogram_widget->xAxis->setLabel("intensity");
    this->ui->histogram_widget->xAxis->setNumberPrecision(8);
    this->ui->histogram_widget->xAxis->setOffset(0);
    this->ui->histogram_widget->xAxis->setPadding(0);
    this->ui->histogram_widget->xAxis->setAntialiased(true);

    this->ui->histogram_widget->yAxis->setLabel("probability");
    this->ui->histogram_widget->yAxis->setOffset(0);
    this->ui->histogram_widget->yAxis->setPadding(0);
    this->ui->histogram_widget->yAxis->setAntialiased(true);

    this->ui->histogram_widget->clearGraphs();
    QCPGraph* graph = this->ui->histogram_widget->addGraph();
    graph->setPen(QPen(QColor(116,205,122)));
    graph->setLineStyle(QCPGraph::lsStepCenter);
    graph->setErrorType(QCPGraph::etValue);

    QVector<double> intensitiesQ = QVector<double>::fromStdVector(intensities);
    QVector<double> probabilitiesQ = QVector<double>::fromStdVector(probabilities);
    graph->setData(intensitiesQ, probabilitiesQ);


    this->ui->histogram_widget->rescaleAxes();
    this->ui->histogram_widget->replot();
}

void ImageWidget::on_histogram_bin_count_spinbox_editingFinished()
{
}

void ImageWidget::on_update_histogram_button_clicked()
{
    this->calculateHistogram();
}

void ImageWidget::on_update_window_spinbox_clicked()
{
    this->paintImage();
    this->calculateHistogram();
}

void ImageWidget::mouseReleaseEvent(QMouseEvent *)
{
    if(this->adding_reference_roi)
    {
        this->adding_reference_roi = false;

        // add roi
    }
}

void ImageWidget::mousePressEvent(QMouseEvent * mouse_event)
{
    if(this->image.IsNull())
        return;

    if(this->inner_image_frame == nullptr || this->show_pixel_value_at_cursor == false ||
            this->ui->operations_panel->isVisible() == false)
        return;

    QPoint position = this->inner_image_frame->mapFromGlobal(mouse_event->globalPos());
    std::cout << "mouse pressed at " << position.x() << "|" << position.y() << std::endl;


    if(this->ui->region_growing_segmentation_widget->isAddingSeedPoint())
    {
        Image::IndexType image_index;
        image_index[0] = position.x();
        image_index[1] = position.y();
        if(InputDimension > 2)
            image_index[2] = this->slice_index;
        this->ui->region_growing_segmentation_widget->addSeedPointAt(image_index);
    }

    int index = this->selectedProfileLineIndex();
    if( index == -1)
    {
        this->ui->status_bar->setText("add a profile line first and select it...");
        return;
    }
    ProfileLine line = this->profile_lines.at(index);

    bool is_left_button = mouse_event->button() == Qt::LeftButton;
    if(is_left_button)
    {
        line.setPosition1(position);
    }
    else
    {
        line.setPosition2(position);
    }
    this->profile_lines[index] = line;
    this->ui->line_profile_list_widget->item(index)->setText(line.text());
    this->paintSelectedProfileLine();
    this->repaint();

    emit this->profileLinesChanged();
}

bool ImageWidget::eventFilter(QObject *target, QEvent *event)
{
    if(this->image.IsNull())
        return false;

    if(event->type() == QEvent::Paint)
        return false;

    if(event->type() == QEvent::MouseMove && target == inner_image_frame)
    {
        QMouseEvent* mouse_event = static_cast<QMouseEvent*>(event);
        if(mouse_event == nullptr || this->show_pixel_value_at_cursor == false)
            return false;

        QPoint position = this->inner_image_frame->mapFromGlobal(mouse_event->globalPos());
       // std::cout << "mouse move at " << position.x() << "|" << position.y() << std::endl;

        Image::SizeType size = this->image->GetLargestPossibleRegion().GetSize();
        if(position.x() < 0 || position.x() > size[0] ||
           position.y() < 0 || position.y() > size[1] )
        {
            return false;
        }
        Image::IndexType index;
        index[0] = position.x();
        index[1] = position.y();
        if(InputDimension > 2)
            index[2] = this->slice_index;

        // showing pixel value...
        Image::PixelType pixel_value = this->image->GetPixel(index);
        this->setPixelInfo(position, pixel_value);

        bool is_left_button = (mouse_event->buttons() & Qt::LeftButton == Qt::LeftButton);

        int selected_roi_index = this->selectedReferenceROI();

        if(this->adding_reference_roi &&
                is_left_button &&
                selected_roi_index > -1)
        {
            this->reference_rois[selected_roi_index].push_back(position);
            this->updateReferenceROI();
        }
    }
    return false; // always returning false, so the pixmap is painted
}

void ImageWidget::setPixelInfo(QPoint position, double pixel_value)
{
    QString text = QString("pixel value at ") +
                           QString::number(position.x()) +
                           " | " +
                           QString::number(position.y()) +
                           " = " +
                           QString::number(pixel_value);
    this->ui->status_bar->setText(text);
}

void ImageWidget::histogram_mouse_move(QMouseEvent* event)
{
    if(this->image.IsNull())
        return;

    QPoint position = event->pos();
    double pixel_value = this->ui->histogram_widget->xAxis->pixelToCoord(position.x());

    this->setPixelInfo(position, pixel_value);
}

void ImageWidget::line_profile_mouse_move(QMouseEvent* event)
{
    if(this->image.IsNull())
        return;

    QPoint position = event->pos();
    double pixel_value = this->ui->line_profile_widget->yAxis->pixelToCoord(position.y());

    this->setPixelInfo(position, pixel_value);
}

void ImageWidget::info_box_toggled(bool show_info_box)
{
    if(show_info_box)
    {
        this->showStatistics();
    }
    else
    {
        this->hideStatistics();
    }
}

void ImageWidget::on_histogram_box_outer_toggled(bool show)
{
    if(show)
    {
        this->showHistogram();
    }
    else
    {
        this->hideHistogram();
    }

}

void ImageWidget::on_add_profile_line_button_clicked()
{
    ProfileLine line;
    this->profile_lines.push_back(line);
    this->ui->line_profile_list_widget->addItem(line.text());
    this->ui->line_profile_list_widget->item(
                this->profile_lines.size() - 1)->setSelected(true);
}

void ImageWidget::paintSelectedProfileLine()
{
    if(this->image.IsNull())
        return;

    int selected_index = this->selectedProfileLineIndex();
    if(selected_index == -1)
    {
        return;
    }
    ProfileLine line = this->profile_lines.at(selected_index);
    if(!line.isSet())
    {
        return;
    }

    std::vector<double> intensities;
    std::vector<double> distances;
    ITKImageProcessor::intensity_profile(this->image,
                                         line.position1().x(),
                                         line.position1().y(),
                                         line.position2().x(),
                                         line.position2().y(),
                                         intensities,
                                         distances);
    this->ui->line_profile_widget->clearGraphs();

    QCPGraph *graph = this->ui->line_profile_widget->addGraph();

    QVector<double> intensitiesQ = QVector<double>::fromStdVector(intensities);
    QVector<double> distancesQ = QVector<double>::fromStdVector(distances);
    graph->setData(distancesQ, intensitiesQ);

    graph->setPen(QPen(Qt::blue));
    graph->setLineStyle(QCPGraph::lsLine);
    graph->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 3));

    this->ui->line_profile_widget->xAxis->setLabel("distance");
    this->ui->line_profile_widget->yAxis->setLabel("intensity");

    this->ui->line_profile_widget->rescaleAxes();
    this->ui->line_profile_widget->replot();
}

void ImageWidget::on_line_profile_list_widget_itemSelectionChanged()
{
    this->paintSelectedProfileLine();
    this->paintImage();
    emit this->selectedProfileLineIndexChanged(this->getSelectedProfileLineIndex());
}

int ImageWidget::selectedProfileLineIndex()
{
    if(this->ui->line_profile_list_widget->selectionModel()->selectedIndexes().size() == 0)
    {
        return -1;
    }
    return this->ui->line_profile_list_widget->selectionModel()->selectedIndexes().at(0).row();
}

int ImageWidget::selectedReferenceROI()
{
    if(this->ui->referenceROIsListWidget->selectionModel()->selectedIndexes().size() == 0)
    {
        return -1;
    }
    return this->ui->referenceROIsListWidget->selectionModel()->selectedIndexes().at(0).row();
}


void ImageWidget::paintSelectedReferenceROI()
{
    if(this->image.IsNull())
        return;

    int index = this->selectedReferenceROI();
    if(index == -1)
    {
        return;
    }
    QVector<QPoint> roi = this->reference_rois[index];

    if(roi.size() == 0)
        return;

    if(inner_image_frame == nullptr)
        return;

    QPolygon polygon(roi);

    QPixmap image = QPixmap::fromImage(*q_image);
    QPainter painter(&image);

    QPen pen(Qt::black);
    pen.setWidth(1);
    painter.setPen(pen);

    QColor color(0,250,0,100);
    QBrush brush(color);
    painter.setBrush(brush);

    painter.drawPolygon(polygon);

    inner_image_frame->setPixmap(image);

}

void ImageWidget::paintSelectedProfileLineInImage()
{
    if(this->image.IsNull())
        return;

    int selected_profile_line_index = this->selectedProfileLineIndex();
    if(selected_profile_line_index == -1)
    {
        return;
    }
    ProfileLine line = this->profile_lines.at(selected_profile_line_index);
    if(!line.isSet())
    {
        return;
    }

    if(inner_image_frame == nullptr)
        return;

    QPixmap image = QPixmap::fromImage(*q_image);
    QPainter painter(&image);

    QPen pen(Qt::blue);
    pen.setWidth(1);
    painter.setPen(pen);
    painter.drawLine(line.position1(), line.position2());

    painter.setPen(QPen(Qt::red,2));
    painter.drawPoint(line.position1());
    painter.setPen(QPen(Qt::green,2));
    painter.drawPoint(line.position2());

    inner_image_frame->setPixmap(image);
}

void ImageWidget::saveImageState()
{
    if(this->image.IsNull())
        return;

    if(this->image_save.IsNull())
    {
        std::cout << "cloning image state" << std::endl;
        this->image_save = this->image; // saving only one time
    }
}

void ImageWidget::restoreImageState()
{
    std::cout << "restoring the cloned image state" << std::endl;
    this->setImage(this->image_save);

}

void ImageWidget::shrink(
            unsigned int shrink_factor_x,
            unsigned int shrink_factor_y,
            unsigned int shrink_factor_z)
{
    if(this->image.IsNull())
        return;

    typedef itk::ShrinkImageFilter<Image, Image> Shrinker;
    typename Shrinker::Pointer shrinker = Shrinker::New();
    shrinker->SetInput( this->image );
    shrinker->SetShrinkFactor(0, shrink_factor_x);
    shrinker->SetShrinkFactor(1, shrink_factor_y);
    shrinker->SetShrinkFactor(2, shrink_factor_z);

    shrinker->Update();
    this->setImage(shrinker->GetOutput());

  //  this->image->SetOrigin( );
  //  this->image->SetSpacing( );
}

void ImageWidget::on_shrink_button_clicked()
{
    if(this->image.IsNull())
        return;

    unsigned int shrink_factor_x = this->ui->shrink_factor_x->text().toUInt();
    unsigned int shrink_factor_y = this->ui->shrink_factor_y->text().toUInt();
    unsigned int shrink_factor_z = this->ui->shrink_factor_z->text().toUInt();

    this->saveImageState();
    this->shrink(shrink_factor_x, shrink_factor_y, shrink_factor_z);
}

void ImageWidget::on_restore_original_button_clicked()
{
    this->restoreImageState();
}


void ImageWidget::displayOriginAndSpacing()
{
    if(this->image.IsNull())
        return;

    Image::PointType origin = this->image->GetOrigin();
    QString origin_text = QString("%1 | %2 | %3").arg(
            QString::number(origin[0], 'g', 3),
            QString::number(origin[1], 'g', 3),
            QString::number(origin[2], 'g', 3)
                );
    this->ui->origin_label->setText(origin_text);

    Image::SpacingType spacing = this->image->GetSpacing();
    QString spacing_text = QString("%1 | %2 | %3").arg(
            QString::number(spacing[0], 'g', 3),
            QString::number(spacing[1], 'g', 3),
            QString::number(spacing[2], 'g', 3)
                );
    this->ui->spacing_label->setText(spacing_text);
}

void ImageWidget::on_load_button_clicked()
{
    QString file_name = QFileDialog::getOpenFileName(this, "open volume file");
    if(file_name == QString::null || !QFile(file_name).exists())
    {
        return;
    }
    Image::Pointer image = ITKImageProcessor::read(file_name.toStdString());
    this->setImage(image);
}

void ImageWidget::on_save_button_clicked()
{
    if(this->image.IsNull())
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save volume file");
    if(file_name.isNull())
    {
        return;
    }
    ITKImageProcessor::write(this->image, file_name.toStdString());
}

void ImageWidget::setInputRanges()
{
    if(this->image.IsNull())
        return;

    Image::RegionType region = image->GetLargestPossibleRegion();
    Image::SizeType size = region.GetSize();

    this->ui->slice_slider->setMinimum(0); // first slice gets slice index 0
    this->ui->slice_spinbox->setMinimum(this->ui->slice_slider->minimum());

    int max_z = 0;
    if(size.GetSizeDimension() >= 3)
    {
        this->ui->slice_slider->setMaximum(size[2] - 1);
        this->ui->slice_spinbox->setMaximum(this->ui->slice_slider->maximum());
        max_z = size[2] - 1;
    }
    int max_x = size[0] - 1;
    int max_y = size[1] - 1;
    this->ui->from_x_spinbox->setMaximum(max_x);
    this->ui->from_y_spinbox->setMaximum(max_y);
    this->ui->from_z_spinbox->setMaximum(max_z);
    this->ui->from_x_spinbox->setValue(0);
    this->ui->from_y_spinbox->setValue(0);
    this->ui->from_z_spinbox->setValue(0);
    this->ui->to_x_spinbox->setMaximum(max_x);
    this->ui->to_y_spinbox->setMaximum(max_y);
    this->ui->to_z_spinbox->setMaximum(max_z);
    this->ui->to_x_spinbox->setValue(max_x);
    this->ui->to_y_spinbox->setValue(max_y);
    this->ui->to_z_spinbox->setValue(max_z);
    this->updateExtractedSizeLabel(7);

}

void ImageWidget::on_extract_button_clicked()
{
    if(this->image.IsNull())
        return;

    Image::Pointer extracted_volume = ITKImageProcessor::extract_volume(
                this->image,
                ui->from_x_spinbox->value(),
                ui->to_x_spinbox->value(),
                ui->from_y_spinbox->value(),
                ui->to_y_spinbox->value(),
                ui->from_z_spinbox->value(),
                ui->to_z_spinbox->value()
                );
    this->saveImageState();
    this->setImage(extracted_volume);
}

void ImageWidget::updateExtractedSizeLabel(int)
{
    int from_x = ui->from_x_spinbox->value();
    int to_x = ui->to_x_spinbox->value();
    int size_x = to_x - from_x + 1;

    int from_y = ui->from_y_spinbox->value();
    int to_y = ui->to_y_spinbox->value();
    int size_y = to_y - from_y + 1;

    int from_z = ui->from_z_spinbox->value();
    int to_z = ui->to_z_spinbox->value();
    int size_z = to_z - from_z + 1;

    QString size_text = QString("%1x%2x%3").arg(
                QString::number(size_x),
                QString::number(size_y),
                QString::number(size_z));

    this->ui->extracted_region_size_label->setText(size_text);
}

void ImageWidget::on_from_x_spinbox_editingFinished()
{

}

void ImageWidget::on_restore_original_button_extract_clicked()
{
    this->restoreImageState();
}

void ImageWidget::on_slice_spinbox_valueChanged(int slice_index)
{
    if(slice_index != this->slice_index)
    {
        this->setSliceIndex(slice_index);
    }
}


void ImageWidget::connectedProfileLinesChanged()
{
    if(this->profile_line_parent == nullptr)
    {
        return;
    }

    this->profile_lines = this->profile_line_parent->getProfileLines();
    this->ui->line_profile_list_widget->clear();
    int index = 0;
    int selected_index = this->profile_line_parent->getSelectedProfileLineIndex();
    for(ProfileLine line : this->profile_lines)
    {
        QListWidgetItem* item = new QListWidgetItem(line.text());
        this->ui->line_profile_list_widget->addItem(item);
        if(index++ == selected_index)
        {
            item->setSelected(true);
        }
    }
    this->paintSelectedProfileLine();
    this->paintImage(); // will call this->paintSelectedProfileLineInImage();
}

void ImageWidget::connectedSelectedProfileLineIndexChanged(int selected_index)
{
    if(selected_index > -1 &&
            this->ui->line_profile_list_widget->model()->rowCount() > selected_index)
    {
        this->ui->line_profile_list_widget->item(selected_index)->setSelected(true);
    }
}
int ImageWidget::getSelectedProfileLineIndex()
{
    return this->selectedProfileLineIndex();
}

void ImageWidget::on_fromMinimumButton_clicked()
{
    bool ok;
    float minimum_pixel_value = this->ui->minimum_label->text().toFloat(&ok);
    if(ok)
    {
        this->ui->window_from_spinbox->setValue(minimum_pixel_value);

        this->paintImage();
        this->calculateHistogram();
    }
}

void ImageWidget::on_toMaximumButton_clicked()
{
    bool ok;
    float maximum_pixel_value = this->ui->maximum_label->text().toFloat(&ok);
    if(ok)
    {
        this->ui->window_to_spinbox->setValue(maximum_pixel_value);

        this->paintImage();
        this->calculateHistogram();
    }
}

void ImageWidget::on_calculate_button_clicked()
{
    if(this->image.IsNull())
        return;

    Image::Pointer image = this->image;

    qint64 start_timestamp = QDateTime::currentMSecsSinceEpoch();
    auto finished_callback = [this, start_timestamp, image] (
            Image::Pointer reflectance)
    {
        qint64 finish_timestamp = QDateTime::currentMSecsSinceEpoch();
        qint64 duration = finish_timestamp - start_timestamp;

        QString text = "finished, duration: " + QString::number(duration) + "ms";
        std::cout << text.toStdString() << std::endl << std::flush;
        emit this->fireStatusTextChange(text);

        ImageWidget* target_widget = this->output_widget == nullptr ? this : this->output_widget;
        emit target_widget->fireImageChange(reflectance);

    };

    if(worker_thread != nullptr)
    {
        this->ui->status_bar->setText("wait for the current worker thread to end");
        return;
    }
    this->worker_thread = new std::thread([=]() {

        ITKImageProcessor::multiScaleRetinex( image,
                                              this->multi_scale_retinex.scales,
                                              finished_callback);

        emit this->fireWorkerFinished();
    });

}



void ImageWidget::showImageOnly()
{
    this->hideHistogram();
    this->hideStatistics();
    this->ui->load_save_panel->hide();
    this->ui->operations_panel->hide();
    this->setMinimumSizeToImage();
}

void ImageWidget::setMinimumSizeToImage()
{
    const int border = 50;
    this->ui->image_frame->setMinimumSize(this->image->GetLargestPossibleRegion().GetSize()[0] + border*2,
            this->image->GetLargestPossibleRegion().GetSize()[1]+border*2);
}

void ImageWidget::hidePixelValueAtCursor()
{
    this->show_pixel_value_at_cursor = false;
}

void ImageWidget::showPixelValueAtCursor()
{
    this->show_pixel_value_at_cursor = true;
}

void ImageWidget::setOutputWidget(ImageWidget* output_widget)
{
    this->output_widget = output_widget;
    this->ui->region_growing_segmentation_widget->setTargetImageWidget(output_widget);
}

void ImageWidget::setOutputWidget2(ImageWidget* output_widget)
{
    this->output_widget2 = output_widget;
}

void ImageWidget::setOutputWidget3(ImageWidget* output_widget)
{
    this->output_widget3 = output_widget;
}

void ImageWidget::paintEvent(QPaintEvent *paint_event)
{
    QWidget::paintEvent(paint_event);
    this->paintImage();
}


void ImageWidget::handleWorkerFinished()
{
    if(this->worker_thread != nullptr)
    {
        this->worker_thread->join();
        delete this->worker_thread;
        this->worker_thread = nullptr;
    }
}

void ImageWidget::handleStatusTextChange(QString text)
{
    this->ui->status_bar->setText(text);
    this->ui->status_bar->repaint();
}

void ImageWidget::handleImageChange(Image::Pointer image)
{
    this->setImage(image);
}

void ImageWidget::setPage(unsigned char page_index)
{
    this->ui->operations_panel->setCurrentIndex(page_index);
}

void ImageWidget::on_pushButton_clicked()
{
    if(this->image.IsNull())
        return;

    float sigma_spatial_distance = this->ui->sigmaSpatialDistanceSpinbox->value();
    float sigma_intensity_distance = this->ui->sigmaIntensityDistanceSpinbox->value();
    int kernel_size = this->ui->kernelSizeSpinbox->value();
    Image::Pointer filtered_image = ITKImageProcessor::bilateralFilter( image,
                                                sigma_spatial_distance,
                                                sigma_intensity_distance,
                                                kernel_size);

    ImageWidget* target_widget = this->output_widget == nullptr ? this : this->output_widget;
    target_widget->setImage(filtered_image);
}

void ImageWidget::on_thresholdButton_clicked()
{
    if(this->image.IsNull())
        return;

    Image::PixelType lower_threshold_value = this->ui->lowerThresholdSpinbox->value();
    Image::PixelType upper_threshold_value = this->ui->upperThresholdSpinbox->value();
    Image::PixelType outside_pixel_value = this->ui->outsideSpinbox->value();
    Image::Pointer filtered_image = ITKImageProcessor::threshold(image,
                                                lower_threshold_value,
                                                upper_threshold_value,
                                                outside_pixel_value);

    ImageWidget* target_widget = this->output_widget == nullptr ? this : this->output_widget;
    target_widget->setImage(filtered_image);


}

void ImageWidget::on_addScaleButton_clicked()
{
    this->multi_scale_retinex.addScaleTo(this->ui->multiScaleRetinexScalesFrame);

}

void ImageWidget::on_pushButton_3_clicked()
{

}

void ImageWidget::on_pushButton_4_clicked()
{
    this->adding_reference_roi = true;

    QVector<QPoint> roi;
    reference_rois.push_back(roi);

    uint index = reference_rois.size() - 1;

    QString item = "ROI" + QString::number(index);
    this->ui->referenceROIsListWidget->addItem(item);
    this->ui->referenceROIsListWidget->setCurrentRow(index);

    this->reference_rois_statistic.push_back(ITKImageProcessor::ReferenceROIStatistic());
}

void ImageWidget::updateReferenceROI()
{
    int index = this->selectedReferenceROI();
    if(index == -1)
        return;

    if(inner_image_frame == nullptr)
        return;

    QVector<QPoint> roi = this->reference_rois[index];

    if(roi.size() == 0)
        return;

    // get mean/median of pixels which are inside the polygon
    QPolygon polygon(roi);

    Image::PixelType window_from = 0;
    Image::PixelType window_to = 0;
    this->userWindow(window_from, window_to);
    QImage *data = ITKToQImageConverter::convert(this->image,
                                                   this->slice_index,
                                                   window_from,
                                                   window_to);

    QList<float> pixelsInside;
    for(int x = 0; x < data->width(); x++)
    {
        for(int y = 0; y < data->height(); y++)
        {
            QPoint point(x,y);
            if(!polygon.containsPoint(point, Qt::OddEvenFill))
                continue;

            QColor color = QColor(data->pixel(x,y));
            float pixel = color.red() / 255.0f;
            pixelsInside.push_back(pixel);
        }
    }
    if(pixelsInside.length() == 0)
        return;

    qSort(pixelsInside);
    uint median_index = pixelsInside.length() / 2;
    float median_value = pixelsInside[median_index];

    QPoint center = polygon.boundingRect().center();

    QString text = QString("Position: (%1, %2), Median: %3").arg(
                QString::number(center.x()),
                QString::number(center.y()),
                QString::number(median_value));
    this->ui->referenceROIsListWidget->item(index)->setText(text);

    ITKImageProcessor::ReferenceROIStatistic& statistic = this->reference_rois_statistic[index];
    statistic.median_value = median_value;
    statistic.x = center.x();
    statistic.y = center.y();
}

void ImageWidget::on_referenceROIsListWidget_currentRowChanged(int currentRow)
{
}

void ImageWidget::on_referenceROIsListWidget_itemSelectionChanged()
{
    this->repaint();

}

void ImageWidget::on_pushButton_6_clicked()
{
    if(this->reference_rois_statistic.size() == 0)
    {
        this->handleStatusTextChange("add some rois first");
        return;
    }


    uint spline_order = this->ui->splineOrderSpinbox->value();
    uint spline_levels = this->ui->splineLevelsSpinbox->value();
    uint spline_control_points = this->ui->splineControlPoints->value();

    if(spline_control_points <= spline_order)
    {
        this->handleStatusTextChange("need more control points than the spline order is");
        return;
    }

    Image::Pointer field_image = nullptr;
    Image::Pointer output_image = ITKImageProcessor::splineFit(
                this->image, spline_order, spline_levels, spline_control_points,
                this->reference_rois_statistic, field_image);

    ImageWidget* target_widget = this->output_widget == nullptr ? this : this->output_widget;
    target_widget->setImage(output_image);

    ImageWidget* target_widget2 = this->output_widget2 == nullptr ? this : this->output_widget2;
    target_widget2->setImage(field_image);

    ITKImageProcessor::printMetric(this->reference_rois_statistic);
    target_widget->setReferenceROIs(this->reference_rois);
}


void ImageWidget::setReferenceROIs(QList<QVector<QPoint>> reference_rois)
{
    this->ui->referenceROIsListWidget->clear();
    this->reference_rois_statistic.clear();

    this->reference_rois = reference_rois;

    for(int i = 0; i < reference_rois.size(); i++)
    {
        QString item = "ROI" + QString::number(i);
        this->ui->referenceROIsListWidget->addItem(item);
        this->ui->referenceROIsListWidget->setCurrentRow(i);
        this->reference_rois_statistic.push_back(ITKImageProcessor::ReferenceROIStatistic());

        updateReferenceROI();
    }

    ITKImageProcessor::printMetric(this->reference_rois_statistic);
}
