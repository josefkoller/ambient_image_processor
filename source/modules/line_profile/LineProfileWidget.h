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

#ifndef LINEPROFILEWIDGET_H
#define LINEPROFILEWIDGET_H

#include <QWidget>
#include "BaseModuleWidget.h"
#include "LineProfile.h"

namespace Ui {
class LineProfileWidget;
}

class LineProfileWidget : public BaseModuleWidget
{
    Q_OBJECT

public:
    explicit LineProfileWidget(QString title, QWidget *parent = 0);
    ~LineProfileWidget();

    QList<LineProfile> getProfileLines() { return this->profile_lines; }
    int selectedProfileLineIndex();

    void mousePressedOnImage(Qt::MouseButton button, ITKImage::Index index);
    void mouseMoveOnImage(Qt::MouseButtons button, ITKImage::Index cursor_index);

    void connectTo(LineProfileWidget* other);
    void paintSelectedProfileLine();
private:
    Ui::LineProfileWidget *ui;

    LineProfile::Point cursor_position;
    QPointF projected_cursor_point;

    QList<LineProfile> profile_lines;

    LineProfileWidget* profile_line_parent;

    ITKImage image;

    bool setting_line_point;

    QVector<double> intensitiesQ;
    QVector<double> distancesQ;
private slots:
    void line_profile_mouse_move(QMouseEvent*);
    void on_add_profile_line_button_clicked();

    void on_line_profile_list_widget_itemSelectionChanged();

    void paintSelectedProfileLineInImage(QPixmap* pixmap);
    void on_setting_line_point_button_clicked();

    void on_connected_to_parent_checkbox_clicked();

signals:
    void profileLinesChanged();
public slots:
    void connectedProfileLinesChanged();

public:
    virtual void registerModule(ImageWidget* image_widget);
    virtual void connectTo(BaseModule* other);

    void addLineProfile(LineProfile line);
    void clearLineProfiles();
private:
    static const QColor line_color;
    static const QColor second_line_color;
    static const QColor cursor_color;
    static const QColor start_point_color;
    static const QColor end_point_color;
    static const QColor line_with_parent_cursor_color;
    static const uint line_profile_width;

public:
    void save_to_file(QString file_name = "");
};

#endif // LINEPROFILEWIDGET_H
