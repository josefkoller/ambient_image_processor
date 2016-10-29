% transforms the intensity values to the range 0..1
function [normalized_image] = normalize_image(image)
    min_value = min(min(image));
    max_value = max(max(image));
    normalized_image = (image - min_value) ./ (max_value - min_value);
end