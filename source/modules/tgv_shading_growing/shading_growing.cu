
#include "tgv_common.cu"
#include "non_local_gradient_kernel.cu"

#include "PixelIndex.h"

#include <queue>
#include <functional>
#include <iostream>

template<typename Pixel>
using MergeSegments = std::function<void(Pixel search_label, Pixel target_label)>;

#include <map>
template<typename Key, typename Value>
bool contains(std::map<Key, Value> map, Key key)
{
    return map.find(key) != map.end();
}

template<typename Pixel>
struct SeedPoint
{
    PixelIndex index;
    Pixel label;
};

template<typename Pixel>
using MaxRecursionDepthReached = std::function<void(SeedPoint<Pixel> point)>;

template<typename Pixel>
__global__ void find_seed_points_kernel(
        Pixel* gradient_magnitude,
        Pixel lower_threshold, Pixel upper_threshold,
        Pixel* label_image,
        const uint voxel_count)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= voxel_count)
        return;

    if(gradient_magnitude[index] > lower_threshold &&
       gradient_magnitude[index] < upper_threshold)
        label_image[index] = index + 1;
    else
        label_image[index] = 0;
}

template<typename Pixel>
__global__ void merge_segments_kernel(
        Pixel* label_image,
        Pixel search_label, Pixel target_label,
        const uint voxel_count)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if(index >= voxel_count)
        return;

    if(label_image[index] == search_label)
        label_image[index] = target_label;
}

template<typename Pixel>
void grow(
        Pixel* gradient_magnitude,
        PixelIndex size,
        Pixel* label_image,
        Pixel current_label,
        const PixelIndex& index,
        Pixel tolerance,
        MergeSegments<Pixel> merge_segments,
        uint recursion_depth,
        uint max_recursion_depth,
        MaxRecursionDepthReached<Pixel> max_recursion_depth_reached)
{
    if(recursion_depth > max_recursion_depth) {
        max_recursion_depth_reached({index, current_label});
        return;
    }
    recursion_depth++;

    label_image[index.linearIndex(size)] = current_label;

    auto neighbouring_indices = index.collectNeighbours(size);
    for(auto neighbouring_index : neighbouring_indices)
    {
        auto neighbouring_linear_index = neighbouring_index.linearIndex(size);
        if(gradient_magnitude[neighbouring_linear_index] > tolerance)
            continue;

        auto neighbour_label = label_image[neighbouring_linear_index];
        if(neighbour_label < 1e-5) // unlabeled
        {
            grow(gradient_magnitude,
                 size,
                 label_image, current_label,
                 neighbouring_index,
                 tolerance,
                 merge_segments,
                 recursion_depth, max_recursion_depth, max_recursion_depth_reached);
        }
        else if(neighbour_label != current_label) // this neighbour already is labeled differently
        {
            merge_segments(neighbour_label, current_label);
        }
    }
}

template<typename Pixel>
void calculate_mean_of_segments_and_target(
        Pixel* label_image,
        Pixel* f,
        Pixel* u_target,
        uint voxel_count)
{
    // mean
    std::map<uint, Pixel> label_sums;
    std::map<uint, uint> label_counts;
    for(uint i = 0; i < voxel_count; i++)
    {
        uint label = label_image[i];
        if(label > 0)
        {
            if(contains(label_counts, label))
            {
                label_counts[label] ++;
                label_sums[label] += f[i];
            }
            else
            {
                label_counts[label] = 1;
                label_sums[label] = f[i];
            }
        }
    }

    for(auto counts_iterator = label_counts.begin(); counts_iterator != label_counts.end(); counts_iterator++)
    {
        uint label = counts_iterator->first;
        uint count = counts_iterator->second;
        label_sums[label] /= count; // writing means to sums
    }
    for(uint i = 0; i < voxel_count; i++)
    {
        uint label = label_image[i];
        u_target[i] = label > 0 ?
                    label_sums[label] : // labeled pixel gets the mean of the segment
                    f[i];                // unlabeled pixel
    }
}

template<typename Pixel>
void shading_growing(Pixel* f,
                     PixelIndex size,
                     Pixel lower_threshold,
                     Pixel upper_threshold,
                     Pixel* u_target,
                     Pixel* label_image,

                     Pixel* non_local_gradient_kernel_data,
                     uint non_local_gradient_kernel_size,
                     Pixel* gradient_x,
                     dim3 block_dimension,
                     dim3 grid_dimension,
                     dim3 grid_dimension_x,
                     dim3 grid_dimension_y,
                     dim3 grid_dimension_z
                     )
{
    const uint voxel_count = size.x * size.y * size.z;
    // 1. gradient magnitude

    non_local_gradient_kernel<<<grid_dimension, block_dimension>>>(
      f, size.x, size.y, size.z,
      non_local_gradient_kernel_data, non_local_gradient_kernel_size,
      gradient_x);
    cudaCheckError( cudaDeviceSynchronize() );
    Pixel* gradient_magnitude = gradient_x;

    // 2. find seeds
    find_seed_points_kernel<<<grid_dimension, block_dimension>>>(
                     gradient_magnitude,
                     lower_threshold, upper_threshold,
                     label_image,
                     voxel_count);
    cudaCheckError( cudaDeviceSynchronize() );

    // 3. region growing
    std::queue<SeedPoint<Pixel>> seed_point_queue;
    for(uint i = 0; i < voxel_count; i++)
    {
        auto label = label_image[i];
        if(label > 0)
        {
            seed_point_queue.push({ PixelIndex(i, size), label });
            label_image[i] = 0;
        }
    }
    std::cout << "Shading Growing: #seeds=" << seed_point_queue.size() << std::endl;

    const uint max_recursion_depth = 1024;
    MaxRecursionDepthReached<Pixel> max_recursion_depth_reached = [&seed_point_queue] (SeedPoint<Pixel> point) {
        seed_point_queue.push(point);
    };
    MergeSegments<Pixel> merge_segments = [grid_dimension, block_dimension, &label_image, voxel_count](Pixel search_label, Pixel target_label) {
        merge_segments_kernel<<<grid_dimension, block_dimension>>>(
                         label_image,
                         search_label, target_label,
                         voxel_count);
        cudaCheckError( cudaDeviceSynchronize() );
      //  std::cout << "Shading Growing: merged " << search_label << " → " << target_label << std::endl;
    };
    while(!seed_point_queue.empty()) {
        auto seed_point = seed_point_queue.front();
        seed_point_queue.pop();

        auto seed_point_label = label_image[seed_point.index.linearIndex(size)];
        if(seed_point_label > 0) // already in segment
            continue;

        grow(gradient_magnitude,
             size,
             label_image,
             seed_point.label,
             seed_point.index,
             upper_threshold,
             merge_segments,
             0, max_recursion_depth, max_recursion_depth_reached);

    //    std::cout << "Shading Growing: #seeds=" << seed_point_queue.size() << std::endl;
    }

    calculate_mean_of_segments_and_target(label_image, f, u_target, voxel_count);
}
