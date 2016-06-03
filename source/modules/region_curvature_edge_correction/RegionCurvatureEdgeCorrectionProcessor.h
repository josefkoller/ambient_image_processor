#ifndef REGIONCURVATUREEDGECORRECTIONPROCESSOR_H
#define REGIONCURVATUREEDGECORRECTIONPROCESSOR_H

#include "ITKImage.h"
#include "RegionGrowingSegmentationProcessor.h"

#include "Vector3.h"

class RegionCurvatureEdgeCorrectionProcessor
{
private:
    RegionCurvatureEdgeCorrectionProcessor();

    struct Node
    {
        Vector3 position;
        ITKImage::PixelType pixel_value;
        Node(Vector3 position, ITKImage::PixelType pixel_value) : position(position), pixel_value(pixel_value) {}
    };

    static ITKImage curvatureImage(ITKImage image);

    typedef RegionGrowingSegmentationProcessor::SegmentEdgePixelsVector EdgePixelsCollection;
    static EdgePixelsCollection findEdgePixels(ITKImage curvature_image,
                                               ITKImage::Index seed_position,
                                               ITKImage::PixelType tolerance);

    static Vector3 findCentroid(EdgePixelsCollection edge_pixels);
    static void interpolate_edge_pixels(ITKImage image, EdgePixelsCollection edge_pixels, Vector3 centroid,
                                        uint count_of_pixels_to_leave, uint count_of_node_pixels,
                                        uint count_of_pixels_to_generate);
    static void interpolate_edge_pixel(ITKImage image, ITKImage::Index edge_pixel, Vector3 centroid,
                                       uint count_of_pixels_to_leave, uint count_of_node_pixels,
                                       uint count_of_pixels_to_generate);
    static Node interpolate_neighbourhood(
            ITKImage image, ITKImage::IndexType center, Vector3 position, ITKImage::Size size);

    static ITKImage::PixelType neighbour_weight(ITKImage::IndexType index, Vector3 position);
public:
    static ITKImage process(ITKImage image, ITKImage::Index seed_point, ITKImage::PixelType tolerance,
                            uint count_of_pixels_to_leave, uint count_of_node_pixels,
                            uint count_of_pixels_to_generate);
};

#endif // REGIONCURVATUREEDGECORRECTIONPROCESSOR_H
