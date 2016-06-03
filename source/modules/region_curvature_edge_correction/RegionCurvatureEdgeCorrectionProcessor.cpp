#include "RegionCurvatureEdgeCorrectionProcessor.h"

#include "CudaImageOperationsProcessor.h"
#include "RegionGrowingSegmentationProcessor.h"

#include <QuickView.h>

RegionCurvatureEdgeCorrectionProcessor::RegionCurvatureEdgeCorrectionProcessor()
{
}


ITKImage RegionCurvatureEdgeCorrectionProcessor::curvatureImage(ITKImage image)
{
    ITKImage::PixelType laplace_kernel[9];
    laplace_kernel[0] = laplace_kernel[2] = laplace_kernel[6] = laplace_kernel[8] = 0;
    laplace_kernel[1] = laplace_kernel[3] = laplace_kernel[5] = laplace_kernel[7] = -1;
    laplace_kernel[4] = 4;
    return CudaImageOperationsProcessor::convolution3x3(image, laplace_kernel);
}

RegionCurvatureEdgeCorrectionProcessor::EdgePixelsCollection
RegionCurvatureEdgeCorrectionProcessor::findEdgePixels(ITKImage curvature_image,
                                                       ITKImage::Index seed_position,
                                                       ITKImage::PixelType tolerance)
{
    RegionGrowingSegmentationProcessor::EdgePixelsCollection segments_edge_pixels;
    RegionGrowingSegmentation::SeedPoint seed_point(seed_position, tolerance);
    RegionGrowingSegmentation::Segment segment("segment1");
    segment.seed_points.push_back(seed_point);
    RegionGrowingSegmentationProcessor::Segments segments;
    segments.push_back(segment);
    RegionGrowingSegmentationProcessor::process(curvature_image, segments, segments_edge_pixels);
    if(segments_edge_pixels.size() == 0) {
        std::cerr << "no segments edge pixels detected" << std::endl;
        return RegionCurvatureEdgeCorrectionProcessor::EdgePixelsCollection();
    }
    return segments_edge_pixels[0];
}

Vector3 RegionCurvatureEdgeCorrectionProcessor::findCentroid(EdgePixelsCollection edge_pixels)
{
    auto first_index = edge_pixels[0];
    Vector3 centroid = Vector3(first_index);

    for(int i = 1; i < edge_pixels.size(); i++)
    {
        auto index = edge_pixels[i];
        centroid += index;
    }
    return centroid / edge_pixels.size();
}



ITKImage::PixelType RegionCurvatureEdgeCorrectionProcessor::neighbour_weight(ITKImage::IndexType index, Vector3 position)
{
    return 1 / ( (position - Vector3(index)).length() + 1);
}

RegionCurvatureEdgeCorrectionProcessor::Node RegionCurvatureEdgeCorrectionProcessor::interpolate_neighbourhood(
        ITKImage image, ITKImage::IndexType center, Vector3 position, ITKImage::Size size)
{
    std::vector<ITKImage::IndexType> indices = center.collectNeighbours(size);

    typedef ITKImage::PixelType Weight;
    std::vector<Weight> weights;
    Weight weight_sum = 0;
    for(auto index : indices)
    {
        Weight weight = neighbour_weight(index, position);
        weights.push_back(weight);
        weight_sum += weight;
    }

    ITKImage::PixelType pixel_value = 0;
    for(int i = 0; i < indices.size(); i++)
    {
        ITKImage::PixelType value = image.getPixel(indices[i]);
        pixel_value += value * weights[i];
    }
    pixel_value /= weight_sum;
    return Node(position, pixel_value);
}

void RegionCurvatureEdgeCorrectionProcessor::interpolate_edge_pixel(ITKImage image,
                                                                    ITKImage::Index edge_pixel, Vector3 centroid,
                                                                    uint count_of_pixels_to_leave,
                                                                    uint count_of_node_pixels,
                                                                    uint count_of_pixels_to_generate)
{
    auto edge = Vector3(edge_pixel);
    auto edge_to_centroid = centroid - edge;
    auto step = edge_to_centroid / edge_to_centroid.length();

    // collect nodes
    ITKImage::Size size(image.width, image.height, image.depth);
    std::vector<Node> nodes;
    auto position = edge + step * count_of_pixels_to_leave;
    for(int step_index = 0; step_index < count_of_node_pixels; step_index++)
    {
        position += step;
        auto index = position.roundToIndex();

        if(!index.isInside(size))
            continue;

        auto node = interpolate_neighbourhood(image, index, position, size);
        nodes.push_back(node);
    }

    if(nodes.size() < 2)
        return;

    // linear regression
    typedef ITKImage::PixelType Value;
    Value mean_x = (nodes.size()-1) / 2.0;
    Value mean_y = 0;
    for(auto node : nodes)
        mean_y += node.pixel_value;
    mean_y /= nodes.size();

    Value k_numerator = 0;
    Value k_denominator = 0;
    for(int x = 0; x < nodes.size(); x++)
    {
        Value y = nodes[x].pixel_value;
        k_numerator += (x - mean_x) * (y - mean_y);
        k_denominator += (x - mean_x) * (x - mean_x);
    }
    Value k = k_numerator / k_denominator;
    Value d = mean_y - k * mean_x;

    position = edge;
    step = -step;
    for(int step_index = 0; step_index < count_of_pixels_to_generate; step_index++)
    {
        position += step;
        auto index = position.roundToIndex();

        if(!index.isInside(size))
            continue;

        int x = -step_index - 1;
        Value y = k * x + d;

        image.setPixel(index, y);
    }
}

void RegionCurvatureEdgeCorrectionProcessor::interpolate_edge_pixels(ITKImage image,
                                                                     EdgePixelsCollection edge_pixels, Vector3 centroid,
                                                                     uint count_of_pixels_to_leave,
                                                                     uint count_of_node_pixels,
                                                                     uint count_of_pixels_to_generate)
{
    for(auto edge_pixel : edge_pixels)
        interpolate_edge_pixel(image, edge_pixel, centroid,
                                      count_of_pixels_to_leave, count_of_node_pixels,
                                      count_of_pixels_to_generate);
}

ITKImage RegionCurvatureEdgeCorrectionProcessor::process(ITKImage image,
                                                         ITKImage::Index seed_position,
                                                         ITKImage::PixelType tolerance,
                                                         uint count_of_pixels_to_leave,
                                                         uint count_of_node_pixels,
                                                         uint count_of_pixels_to_generate)
{
    // 1. curvature image
    auto curvature_image = curvatureImage(image);

    // 2. find edge pixels
    auto edge_pixels = findEdgePixels(curvature_image, seed_position, tolerance);
    if(edge_pixels.size() == 0) {
        std::cerr << "no edge pixels detected" << std::endl;
        return ITKImage();
    }

    // 3. find centroid
    Vector3 centroid = findCentroid(edge_pixels);

    /*
    // Visualize
    ITKImage edge_pixels_image = image.cloneSameSizeWithZeros();
    for(auto index : edge_pixels)
        edge_pixels_image.setPixel(index, 1.23);
    ITKImage centroid_image = image.cloneSameSizeWithZeros();
    centroid_image.setPixel(centroid.x,centroid.y,centroid.z, 1);
    QuickView quick_view;
    quick_view.AddImage(edge_pixels_image.getPointer().GetPointer(), true, "Edge Pixels");
    quick_view.AddImage(centroid_image.getPointer().GetPointer(), true, "Centroid");
    quick_view.InterpolateOff();
    quick_view.Visualize();
    */

    ITKImage result = image.clone();
    interpolate_edge_pixels(result, edge_pixels, centroid,
                            count_of_pixels_to_leave, count_of_node_pixels,
                            count_of_pixels_to_generate);

    return result;
}
