#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPaintEvent>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(std::string image_path);
    ~MainWindow();

private slots:

private:
    Ui::MainWindow *ui;

    std::string source_image_path;
};

#endif // MAINWINDOW_H
