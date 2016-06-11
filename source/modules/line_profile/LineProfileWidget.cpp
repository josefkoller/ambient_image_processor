#include "LineProfileWidget.h"
#include "ui_LineProfileWidget.h"

#include "LineProfileProcessor.h"

#include <QFileDialog>

const QColor LineProfileWidget::line_with_parent_color = QColor(0, 102, 101);
const QColor LineProfileWidget::line_color = QColor(0, 51, 153);
const QColor LineProfileWidget::cursor_color = QColor(255, 99, 49);
const QColor LineProfileWidget::start_point_color = QColor(202, 0, 50);
const QColor LineProfileWidget::end_point_color = QColor(255, 204, 51);

LineProfileWidget::LineProfileWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::LineProfileWidget),
    profile_line_parent(nullptr),
    setting_line_point(false),
    image(ITKImage::Null)
{
    ui->setupUi(this);

    this->ui->custom_plot_widget->setMouseTracking(true);
    connect(this->ui->custom_plot_widget, &QCustomPlot::mouseMove,
            this, &LineProfileWidget::line_profile_mouse_move);
}

LineProfileWidget::~LineProfileWidget()
{
    delete ui;
}

void LineProfileWidget::line_profile_mouse_move(QMouseEvent* event)
{
    if(this->image.isNull())
        return;

    QPoint position = event->pos();
    double pixel_value = this->ui->custom_plot_widget->yAxis->pixelToCoord(position.y());

    QString text = QString("pixel value at ") +
            QString::number(position.x()) +
            " | " +
            QString::number(position.y()) +
            " = " +
            QString::number(pixel_value);
    this->setStatusText(text);
}


void LineProfileWidget::paintSelectedProfileLine()
{
    this->ui->custom_plot_widget->clearGraphs();

    auto image = this->getSourceImage();
    if(image.isNull())
        return;

    int selected_index = this->selectedProfileLineIndex();
    if(selected_index == -1)
    {
        return;
    }
    LineProfile line = this->profile_lines.at(selected_index);
    if(!line.isSet())
    {
        return;
    }

    std::vector<double> intensities;
    std::vector<double> distances;
    LineProfileProcessor::intensity_profile(image,
                                            line.position1(),
                                            line.position2(),
                                            intensities,
                                            distances);

    if(this->profile_line_parent != nullptr && this->ui->connected_to_parent_checkbox->isChecked())
    {
        QCPGraph *parent_graph = this->ui->custom_plot_widget->addGraph();
        parent_graph->setData(this->profile_line_parent->distancesQ,
                              this->profile_line_parent->intensitiesQ);
        parent_graph->setPen(QPen(line_color,1));
        parent_graph->setLineStyle(QCPGraph::lsLine);
        parent_graph->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 3));
    }

    QCPGraph *graph = this->ui->custom_plot_widget->addGraph();

    this->intensitiesQ = QVector<double>::fromStdVector(intensities);
    this->distancesQ = QVector<double>::fromStdVector(distances);
    graph->setData(distancesQ, intensitiesQ);

    auto pen_color = this->profile_line_parent == nullptr ? line_color : line_with_parent_color;
    graph->setPen(QPen(pen_color));
    graph->setLineStyle(QCPGraph::lsLine);
    graph->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 3));

    this->ui->custom_plot_widget->xAxis->setLabel("distance");
    this->ui->custom_plot_widget->yAxis->setLabel("intensity");

    // cursor position ...

    QPoint line_direction = ITKImage::pointFromIndex(line.position2()) -
                            ITKImage::pointFromIndex(line.position1());
    double line_length = std::sqrt(QPoint::dotProduct(line_direction, line_direction));
    QPointF point1_to_cursor = this->projected_cursor_point - ITKImage::pointFromIndex(line.position1());
    double point1_to_cursor_length = std::sqrt(QPointF::dotProduct(point1_to_cursor, point1_to_cursor));
    double cursor_factor = point1_to_cursor_length / line_length;
    if(cursor_factor >= 0 && cursor_factor <= 1 && distancesQ.size() > 0)
    {
        uint cursor_index = (distancesQ.size()-1) * cursor_factor;

        double cursor_distance = distancesQ[cursor_index];
        double cursor_intensity = intensitiesQ[cursor_index];
        QVector<double> cursor_distance_vector;
        cursor_distance_vector.push_back(cursor_distance);
        QVector<double> cursor_intensity_vector;
        cursor_intensity_vector.push_back(cursor_intensity);
        QCPGraph *graph2 = this->ui->custom_plot_widget->addGraph();
        graph2->setPen(QPen(cursor_color, 2));
        graph2->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 3));
        graph2->setData(cursor_distance_vector, cursor_intensity_vector);
    }
    this->ui->custom_plot_widget->rescaleAxes();
    this->ui->custom_plot_widget->replot();
}


int LineProfileWidget::selectedProfileLineIndex()
{
    if(this->ui->line_profile_list_widget->selectionModel()->selectedIndexes().size() == 0)
    {
        return -1;
    }
    return this->ui->line_profile_list_widget->selectionModel()->selectedIndexes().at(0).row();
}

void LineProfileWidget::mouseMoveOnImage(Qt::MouseButtons button, ITKImage::Index cursor_index)
{
    this->cursor_position = cursor_index;

    if(this->selectedProfileLineIndex() > -1)
        emit this->profileLinesChanged(); // repaint selected profile line in image
}

void LineProfileWidget::mousePressedOnImage(Qt::MouseButton button, ITKImage::Index cursor_index)
{
    if(this->profile_line_parent != nullptr && this->ui->connected_to_parent_checkbox->isChecked())
    {
        this->profile_line_parent->mousePressedOnImage(button, cursor_index);
        return;
    }

    int index = this->selectedProfileLineIndex();
    if( index == -1 || !setting_line_point)
    {
        this->setStatusText("add a profile line first and select it...");
        return;
    }
    LineProfile& line = this->profile_lines[index];

    bool is_left_button = button == Qt::LeftButton;
    if(is_left_button)
    {
        line.setPosition1(cursor_index);
    }
    else
    {
        line.setPosition2(cursor_index);
    }
    this->ui->line_profile_list_widget->item(index)->setText(line.text());

    emit this->profileLinesChanged();
}



void LineProfileWidget::connectedProfileLinesChanged()
{
    if(this->profile_line_parent == nullptr)
        return;

    this->profile_lines = this->profile_line_parent->getProfileLines();
    this->ui->line_profile_list_widget->clear();
    int index = 0;
    int selected_index = this->profile_line_parent->selectedProfileLineIndex();
    for(LineProfile line : this->profile_lines)
    {
        QListWidgetItem* item = new QListWidgetItem(line.text());
        this->ui->line_profile_list_widget->addItem(item);
        if(index++ == selected_index)
        {
            item->setSelected(true);
        }
    }

    this->cursor_position = this->profile_line_parent->cursor_position;

    emit this->profileLinesChanged();
}

void LineProfileWidget::on_add_profile_line_button_clicked()
{
    if(this->profile_line_parent != nullptr && this->ui->connected_to_parent_checkbox->isChecked())
    {
        this->profile_line_parent->on_add_profile_line_button_clicked();
        return;
    }


    LineProfile line;
    this->profile_lines.push_back(line);
    this->ui->line_profile_list_widget->addItem(line.text());
    this->ui->line_profile_list_widget->item(
                this->profile_lines.size() - 1)->setSelected(true);

    this->setting_line_point = true;
    this->ui->setting_line_point_button->setFlat(true);
}

void LineProfileWidget::registerModule(ImageWidget* image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    this->connect(this, &LineProfileWidget::profileLinesChanged,
                  image_widget, &ImageWidget::repaintImageOverlays);

    this->connect(image_widget, &ImageWidget::mousePressedOnImage,
                  this, &LineProfileWidget::mousePressedOnImage);

    this->connect(image_widget, &ImageWidget::mouseMoveOnImage,
                  this, &LineProfileWidget::mouseMoveOnImage);

    connect(image_widget, &ImageWidget::imageChanged,
            this, [this] (ITKImage itk_image) {
        this->image = itk_image;
        emit this->profileLinesChanged();
    });

    connect(image_widget, &ImageWidget::pixmapPainted,
            this, &LineProfileWidget::paintSelectedProfileLineInImage);
}

void LineProfileWidget::on_line_profile_list_widget_itemSelectionChanged()
{
    //   std::cout << "selected profile line: " << this->selectedProfileLineIndex() << std::endl;

    if(this->selectedProfileLineIndex() > -1) {
        this->setting_line_point = true;
        this->ui->setting_line_point_button->setFlat(true);
    }

    emit this->profileLinesChanged();
}


void LineProfileWidget::paintSelectedProfileLineInImage(QPixmap* pixmap)
{
    if(this->image.isNull())
        return;

    int selected_profile_line_index = this->selectedProfileLineIndex();
    if(selected_profile_line_index == -1)
    {
        return;
    }
    LineProfile line = this->getProfileLines().at(selected_profile_line_index);
    if(!line.isSet())
    {
        return;
    }

    QPainter painter(pixmap);

    auto pen_color = this->profile_line_parent == nullptr ? line_color : line_with_parent_color;
    QPen pen(pen_color);
    pen.setWidth(1);
    painter.setPen(pen);

    QPoint point1 = ITKImage::pointFromIndex(line.position1());
    QPoint point2 = ITKImage::pointFromIndex(line.position2());

    painter.drawLine(point1, point2);

    painter.setPen(QPen(start_point_color,2));
    painter.drawPoint(point1);
    painter.setPen(QPen(end_point_color,2));
    painter.drawPoint(point2);

    if(this->image.contains(cursor_position))
    {
        QPointF cursor_point = ITKImage::pointFromIndex(cursor_position);
        QPointF cursor_direction = QPointF(point2.x() - cursor_point.x(),
                                           point2.y() - cursor_point.y());
        QPointF line_direction = QPointF(
                    point2.x() - point1.x(),
                    point2.y() - point1.y());
        float projection = QPointF::dotProduct(line_direction, cursor_direction);
        projection /= QPointF::dotProduct(line_direction, line_direction);
        if(projection >= 0 && projection <= 1)
        {
            this->projected_cursor_point = QPointF(point1) + line_direction * (1 - projection);
            painter.setPen(QPen(cursor_color,2));
            painter.drawPoint(projected_cursor_point);
        }
    }

    this->paintSelectedProfileLine();
}

void LineProfileWidget::connectTo(BaseModule* other)
{
    auto other_line_profile_widget = dynamic_cast<LineProfileWidget*>(other);
    if(other_line_profile_widget == nullptr)
        return;

    this->profile_line_parent = other_line_profile_widget;

    connect(other_line_profile_widget, &LineProfileWidget::profileLinesChanged,
            this, &LineProfileWidget::connectedProfileLinesChanged);
}

void LineProfileWidget::on_setting_line_point_button_clicked()
{
    this->setting_line_point = !this->setting_line_point;
    this->ui->setting_line_point_button->setFlat(this->setting_line_point);
}

void LineProfileWidget::on_connected_to_parent_checkbox_clicked()
{
    this->connectedProfileLinesChanged();
    emit this->profileLinesChanged();
}

void LineProfileWidget::save_to_file()
{
    if(this->image.isNull())
        return;

    QString file_name = QFileDialog::getSaveFileName(this, "save image file with overlays");
    if(file_name.isNull())
        return;

    bool saved = false;
    if(file_name.endsWith("pdf"))
        saved = this->ui->custom_plot_widget->savePdf(file_name);
    if(file_name.endsWith("png"))
        saved = this->ui->custom_plot_widget->savePng(file_name,0,0,1.0, 100);  // 100 ... uncompressed

    this->setStatusText( (saved ? "saved " : "(pdf,png supported) error while saving ") + file_name);
}
