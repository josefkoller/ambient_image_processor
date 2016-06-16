#set dgrid3d 30,30
#set hidden3d
set xlabel x
set ylabel y
set zlabel z
set nokey
#set contour base
#set cntrparam level 10
splot filename using 1:2:3 with points
pause -1