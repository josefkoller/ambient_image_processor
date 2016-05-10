
#include <iostream>
#include "MainWindow.h"
#include <QApplication>

#include <QFile>

int main(int argc, char *argv[])
{
    std::cout << "started program: " << argv[0] << std::endl;
    std::string image_path = "";
    if(argc > 1)
    {
        image_path = argv[1];
        std::cout << "image file path: " << image_path << std::endl;
    }

    if(!QFile(QString::fromStdString(image_path)).exists())
    {
        std::cout << "image file does not exist" << std::endl;
    }

    QApplication application(argc, argv);
    MainWindow window(image_path);
    window.show();

    application.exec();

    std::cout << "finished program: " << argv[0] << std::endl;
    return 0;
}
