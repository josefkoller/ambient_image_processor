close all;
clear all;

pkg load 'image';

field_filename = 'field_z.png';
%field_filename = 'field_linear_x_4_6_256.png';

field = double(imread(field_filename));


 field = normalize_image(field);
 field = 1.0 - field;
% field = field .* 2 - 1;
% field = field .* 128;
% 

folder = '../../../../../test_data/joe_mondrain/joe_mondrain_13/'

image_filename = 'mondrain13.png';
image = imread([folder, image_filename]);

image = double(image);
if size(image, 3) > 1
    image = rgb2gray(image);
end

image = double(image);
image = normalize_image(image);

operator_name = 'add'  % PARAMETER
operator_name = 'multiply'  % PARAMETER

if strcmp(operator_name, 'add')
  image_with_field = image + field; % ADD
else
  image_with_field = image .* field; % MULTIPLICATIVE
end

subplot(431);
imshow(image,[]);
title('image');

subplot(432);
imshow(field,[]);
title('illumination');

subplot(433);
imshow(image_with_field,[]);
title('illuminated image');

output_filename = [folder, image_filename, '_', operator_name, ...
  '_', field_filename]; 

image_with_field = normalize_image(image_with_field);
imwrite(image_with_field, output_filename);


subplot(434);
imhist(mat2gray(image));

subplot(435);
imhist(mat2gray(field));

subplot(436);
imhist(image_with_field);


image_gradient = imgradient(mat2gray(image), 'Sobel');
subplot(437);
imshow(image_gradient,[]);
title('image gradient');

field_gradient = imgradient(mat2gray(field), 'Sobel');
subplot(4,3,8);
imshow(double(field_gradient),[]);
title('illumination');

image_with_field_gradient = imgradient(image_with_field, 'Sobel');
subplot(4,3,9);
imshow(image_with_field_gradient,[]);
title('illuminated image');

subplot(4,3,10);
imhist(image_gradient);

subplot(4,3,11);
imhist(field_gradient);

subplot(4,3,12);
imhist(image_with_field_gradient);

