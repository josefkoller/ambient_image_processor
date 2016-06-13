
#include "stdio.h"

typedef const unsigned int DimensionSize;

template<typename Pixel>
Pixel  mean(Pixel* image, DimensionSize voxel_count)
{
    Pixel sum = 0;
    for(int i = 0; i < voxel_count; i++)
        sum+= image[i];
    return sum / voxel_count;
}

template<typename Pixel>
Pixel  standard_deviation(Pixel* image, DimensionSize voxel_count, Pixel mean)
{
    Pixel sum = 0;
    for(int i = 0; i < voxel_count; i++)
    {
        Pixel difference = image[i] - mean;
        sum+= difference * difference;
    }
    return std::sqrt(sum / (voxel_count - 1));
}

template<typename Pixel>
Pixel  normalized_cross_correlation(
    Pixel* image1, Pixel* image2,
    DimensionSize voxel_count)
{
    const auto mean1 = mean(image1, voxel_count);
    const auto std1 = standard_deviation(image1, voxel_count, mean1);
    const auto mean2 = mean(image2, voxel_count);
    const auto std2 = standard_deviation(image2, voxel_count, mean2);

    printf("mean1: %f \n", mean1);
    printf("std1: %f \n", std1);
    printf("mean2: %f \n", mean2);
    printf("std2: %f \n", std2);

    Pixel normalized_cross_correlation = 0;
    for(int i = 0; i < voxel_count; i++)
    {
        const auto difference1 = image1[i] - mean1;
        const auto difference2 = image2[i] - mean2;
        normalized_cross_correlation+= difference1 * difference2;
    }
    return normalized_cross_correlation / (std1 * std2 * voxel_count);
}

template<typename Pixel>
Pixel  sum_of_absolute_differences(Pixel* image1, Pixel* image2, DimensionSize voxel_count)
{
    Pixel abs_change_sum = 0;
    for(int i = 0; i < voxel_count; i++)
    {
        abs_change_sum+= std::abs(image1[i] - image2[i]);
    }
    return abs_change_sum;
}
