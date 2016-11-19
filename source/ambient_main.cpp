
#include <iostream>
#include "MainWindow.h"
#include <QApplication>

#include <QFile>

int main(int argc, char *argv[])
{
 //   std::cout << "started program: " << argv[0] << std::endl;
    std::string image_path = "";
    if(argc > 1)
    {
        image_path = argv[1];
    }
    if(!QFile(QString::fromStdString(image_path)).exists())
    {
        std::cout << "image file does not exist: " << image_path << std::endl;
    }

    std::string image_path2 = "";
    if(argc > 2)
    {
        image_path2 = argv[2];
    }
    if(!QFile(QString::fromStdString(image_path2)).exists())
    {
        std::cout << "image file does not exist: " << image_path2 << std::endl;
    }

    QApplication application(argc, argv);
    MainWindow window(image_path, image_path2);
    window.show();

    application.exec();

  //  std::cout << "finished program: " << argv[0] << std::endl;
    return 0;
}
