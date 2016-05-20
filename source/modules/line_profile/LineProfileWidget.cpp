#include "LineProfileWidget.h"
#include "ui_LineProfileWidget.h"

#include "LineProfileProcessor.h"

LineProfileWidget::LineProfileWidget(QString title, QWidget *parent) :
    BaseModuleWidget(title, parent),
    ui(new Ui::LineProfileWidget),
    adding_profile_line(false),
    profile_line_parent(nullptr),
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
    auto image = this->getSourceImage();
    if(image.isNull())
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
                                            line.position1().x(),
                                            line.position1().y(),
                                            line.position2().x(),
                                            line.position2().y(),
                                            intensities,
                                            distances);
    this->ui->custom_plot_widget->clearGraphs();

    QCPGraph *graph = this->ui->custom_plot_widget->addGraph();

    QVector<double> intensitiesQ = QVector<double>::fromStdVector(intensities);
    QVector<double> distancesQ = QVector<double>::fromStdVector(distances);
    graph->setData(distancesQ, intensitiesQ);

    graph->setPen(QPen(Qt::blue));
    graph->setLineStyle(QCPGraph::lsLine);
    graph->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 3));

    this->ui->custom_plot_widget->xAxis->setLabel("distance");
    this->ui->custom_plot_widget->yAxis->setLabel("intensity");

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

void LineProfileWidget::mousePressedOnImage(Qt::MouseButton button, QPoint position)
{
    int index = this->selectedProfileLineIndex();
    if( index == -1)
    {
        this->setStatusText("add a profile line first and select it...");
        return;
    }
    LineProfile line = this->profile_lines.at(index);

    bool is_left_button = button == Qt::LeftButton;
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

    emit this->profileLinesChanged();
}



void LineProfileWidget::connectedProfileLinesChanged()
{
    if(this->profile_line_parent == nullptr)
    {
        return;
    }

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

    emit this->profileLinesChanged();
}

void LineProfileWidget::on_add_profile_line_button_clicked()
{
    LineProfile line;
    this->profile_lines.push_back(line);
    this->ui->line_profile_list_widget->addItem(line.text());
    this->ui->line_profile_list_widget->item(
                this->profile_lines.size() - 1)->setSelected(true);
}

void LineProfileWidget::registerModule(ImageWidget* image_widget)
{
    BaseModuleWidget::registerModule(image_widget);

    this->connect(this, &LineProfileWidget::profileLinesChanged,
                  image_widget, &ImageWidget::handleRepaintImage);

    this->connect(image_widget, &ImageWidget::mousePressedOnImage,
                  this, &LineProfileWidget::mousePressedOnImage);

    connect(image_widget, &ImageWidget::imageChanged,
            this, [this] (ITKImage& itk_image) {
        this->image = itk_image;
    });

    connect(image_widget, &ImageWidget::pixmapPainted,
            this, &LineProfileWidget::paintSelectedProfileLineInImage);
}

void LineProfileWidget::on_line_profile_list_widget_itemSelectionChanged()
{
    //   std::cout << "selected profile line: " << this->selectedProfileLineIndex() << std::endl;

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

    QPen pen(Qt::blue);
    pen.setWidth(1);
    painter.setPen(pen);
    painter.drawLine(line.position1(), line.position2());

    painter.setPen(QPen(Qt::red,2));
    painter.drawPoint(line.position1());
    painter.setPen(QPen(Qt::green,2));
    painter.drawPoint(line.position2());

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
