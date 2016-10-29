close all;
clear all;

pkg load 'image';

size = 256;

image = zeros(size, size);
image = double(image);

for x=1:size
  for y=1:size
    v = x;
    
    image(y,x) = v;
  end
end

image_min = min(min(image));
image = (image - image_min) / (max(max(image)) - image_min);


target_min = 0.1;
target_max = 0.9;

target_spectrum = target_max - target_min;
image = image * target_spectrum + target_min;

imshow(image, []);

filename = sprintf('field_linear_x_%d_%d.png', target_min*10, target_max*10);
imwrite(image, filename);

info = imfinfo(filename);
bit_depth = info.BitDepth

