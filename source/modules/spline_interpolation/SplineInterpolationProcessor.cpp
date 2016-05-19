#include "SplineInterpolationProcessor.h"

#include <itkPointSet.h>
#include <itkBSplineScatteredDataPointSetToImageFilter.h>
#include <itkVectorIndexSelectionCastImageFilter.h>
#include <itkDivideImageFilter.h>

SplineInterpolationProcessor::SplineInterpolationProcessor()
{
}


SplineInterpolationProcessor::ImageType::Pointer SplineInterpolationProcessor::process(
        ImageType::Pointer image, uint spline_order, uint spline_levels, uint spline_control_points,
        std::vector<ReferenceROIStatistic> nodes,
        ImageType::Pointer& field_image)
{
    typedef itk::Vector<ImageType::PixelType, 1> ScalarType;
    typedef itk::PointSet<ScalarType, ImageType::ImageDimension> PointSet;
    PointSet::Pointer fieldPoints = PointSet::New();
    fieldPoints->Initialize();

    typedef itk::Image<ScalarType, ImageType::ImageDimension> ScalarImageType;
    typedef typename itk::BSplineScatteredDataPointSetToImageFilter<PointSet, ScalarImageType> BSplineFilter;
    BSplineFilter::WeightsContainerType::Pointer weights = BSplineFilter::WeightsContainerType::New();
    weights->Initialize();

    unsigned int index = 0;
    for(ReferenceROIStatistic node_info : nodes)
    {
        ImageType::IndexType node;
        node[0] = node_info.x;
        node[1] = node_info.y;

        PointSet::PointType point;
        image->TransformIndexToPhysicalPoint(node, point);

        auto pixel_value = image->GetPixel(node);
        ScalarType scalar;
        scalar[0] = pixel_value;

        fieldPoints->SetPointData(index, scalar);
        fieldPoints->SetPoint(index, point);

        weights->InsertElement(index, 1.0);
        ++index;
    }


    typename BSplineFilter::Pointer bspliner = BSplineFilter::New();

    typename BSplineFilter::ArrayType numberOfControlPoints;
    typename BSplineFilter::ArrayType numberOfFittingLevels;
    numberOfControlPoints.Fill(spline_control_points);
    numberOfFittingLevels.Fill(spline_levels);

    ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();

    bspliner->SetOrigin(image->GetOrigin());
    bspliner->SetSpacing(image->GetSpacing());
    bspliner->SetSize(size);
    bspliner->SetDirection(image->GetDirection());
    bspliner->SetGenerateOutputImage( true );
    bspliner->SetNumberOfLevels(numberOfFittingLevels);
    bspliner->SetSplineOrder(spline_order);
    bspliner->SetNumberOfControlPoints(numberOfControlPoints);
    bspliner->SetInput(fieldPoints);
    bspliner->SetPointWeights(weights);
    try
    {
        bspliner->Update();
    }
    catch(itk::ExceptionObject exception)
    {
        exception.Print(std::cout);
        return nullptr;
    }

    typedef itk::VectorIndexSelectionCastImageFilter<ScalarImageType, ImageType> CastFilter;
    typename CastFilter::Pointer cast_filter = CastFilter::New();
    cast_filter->SetInput(bspliner->GetOutput());
    cast_filter->SetIndex(0);
    ImageType::Pointer output = cast_filter->GetOutput();
    output->Update();
    output->DisconnectPipeline();
    output->SetRegions( image->GetRequestedRegion() );
    field_image = output;

    // set zero values to very small
    const ImageType::PixelType MINIMUM_PIXEL_VALUE = 1e-5f;
    itk::ImageRegionIterator<ImageType> iterator(field_image,
                                                 field_image->GetLargestPossibleRegion());
    while(!iterator.IsAtEnd())
    {
        ImageType::PixelType value = iterator.Get();
        if(value < MINIMUM_PIXEL_VALUE)
            iterator.Set(MINIMUM_PIXEL_VALUE);
        ++iterator;
    }

    // divide ...
    typedef itk::DivideImageFilter<ImageType, ImageType, ImageType> DivideFilter;
    DivideFilter::Pointer divide_filter = DivideFilter::New();
    divide_filter->SetInput1(image);
    divide_filter->SetInput2(field_image);
    divide_filter->Update();
    ImageType::Pointer corrected_image = divide_filter->GetOutput();
    corrected_image->DisconnectPipeline();


    return corrected_image;
}


void SplineInterpolationProcessor::printMetric(std::vector<ReferenceROIStatistic> rois)
{
    std::vector<ImageType::PixelType> v;
    std::for_each (std::begin(rois), std::end(rois), [&](const ReferenceROIStatistic roi) {
        v.push_back(roi.median_value);
    });

    double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
    double m =  sum / v.size();

    double accum = 0.0;
    std::for_each (std::begin(v), std::end(v), [&](const double d) {
        accum += (d - m) * (d - m);
    });

    double stdev = sqrt(accum / (v.size()-1));

    std::cout << "mean: " << m << std::endl;
    std::cout << "standard deviation: " << stdev << std::endl;
}
