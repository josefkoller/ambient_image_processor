% based on the formula and parameters in
% 
% S.L. Keeling, M. Hintermüller, F.Knoll, D. Kraft, A. Laurain.
% A Total Variation Based Approach To Correcting Surface Coil 
% Magnet Resonance Images, Applied Mathematics And Computation,
% 2018(2):219-232, 2011

clear;
close all;

Nimage = 256;
spacing = 1.0 / Nimage;
border = 1;
x = 0 : spacing : (border - spacing);
y = x;

mu = 0.01;
nu = 0.01;
zz = 0.5;

n = length(x);
z = zeros(n,n);
for xi = 1:n
    for yi = 1:n
        xf = x(xi);
        yf = y(yi);
        z(xi,yi) = 0.5 * (xf * yf - zz) ^ 2 + 0.5 * (nu * yf ^ 2) + mu * abs(xf);
    end
end

z_image = normalize_image(z);
imwrite(z_image, 'field_z.png');
 
z = -z;


[dx,dy] = gradient(z, spacing, spacing);

figure;
number_of_contour_lines = 32;
contour(x,y,z, number_of_contour_lines);
hold on;

quiver(x,y,dx,dy);

[minimum_value, minimum_index] = min(min(abs(z)));
[minimum_x, minimum_y] = find(abs(z) == minimum_value);

plot(minimum_x / length(x), minimum_y / length(y), 'r*');
hold off;

figure;
imshow(z_image, []);
