unset logscale

stats filename using 3 nooutput name 'Z_'
stats filename using 1 every ::Z_index_min::Z_index_min nooutput
X_min = STATS_min
stats filename using 2 every ::Z_index_min::Z_index_min nooutput
Y_min = STATS_min

if (exists('use_logscale')) \
  set logscale xy \
else \
  set autoscale xy

show logscale

set dgrid3d 10,10
set hidden3d
set xlabel x
set ylabel y
set zlabel z offset 4,-4.5,4
#set nokey
set contour base
set cntrparam level 10

set label at X_min, Y_min, Z_min point pt 7
set label 1 "Minimum" at X_min, Y_min, Z_min offset 1,0.5

splot filename using 1:2:3 with linespoints title z
pause -1
