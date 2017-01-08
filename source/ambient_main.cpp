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
