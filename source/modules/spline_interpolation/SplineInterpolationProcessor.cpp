#include "SplineInterpolationProcessor.h"

#include <itkPointSet.h>
#include <itkBSplineScatteredDataPointSetToImageFilter.h>
#include <itkVectorIndexSelectionCastImageFilter.h>
#include <itkDivideImageFilter.h>
#include <itkStatisticsImageFilter.h>

#include "ExtractProcessor.h"

SplineInterpolationProcessor::SplineInterpolationProcessor()
{
}


ITKImage SplineInterpolationProcessor::process(
        ITKImage image, uint spline_order, uint spline_levels, uint spline_control_points,
        std::vector<ReferenceROIStatistic> nodes,
        ITKImage& field_image)
{
    typedef itk::Vector<ITKImage::PixelType, 1> ScalarType;
    typedef itk::PointSet<ScalarType, ITKImage::ImageDimension> PointSet;
    PointSet::Pointer fieldPoints = PointSet::New();
    fieldPoints->Initialize();

    typedef itk::Image<ScalarType, ITKImage::ImageDimension> ScalarImageType;
    typedef typename itk::BSplineScatteredDataPointSetToImageFilter<PointSet, ScalarImageType> BSplineFilter;
    BSplineFilter::WeightsContainerType::Pointer weights = BSplineFilter::WeightsContainerType::New();
    weights->Initialize();


    typedef ITKImage::InnerITKImage ImageType;
    ImageType::Pointer itk_image = image.getPointer();

    unsigned int index = 0;
    for(ReferenceROIStatistic node_info : nodes)
    {
        ITKImage::InnerITKImage::IndexType node = node_info.point;

        PointSet::PointType point;
        itk_image->TransformIndexToPhysicalPoint(node, point);

        auto pixel_value = itk_image->GetPixel(node);
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

    ITKImage::InnerITKImage::SizeType size = itk_image->GetLargestPossibleRegion().GetSize();

    bspliner->SetOrigin(itk_image->GetOrigin());
    bspliner->SetSpacing(itk_image->GetSpacing());
    bspliner->SetSize(size);
    bspliner->SetDirection(itk_image->GetDirection());
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
        return ITKImage();
    }

    typedef itk::VectorIndexSelectionCastImageFilter<ScalarImageType, ITKImage::InnerITKImage> CastFilter;
    typename CastFilter::Pointer cast_filter = CastFilter::New();
    cast_filter->SetInput(bspliner->GetOutput());
    cast_filter->SetIndex(0);
    ITKImage::InnerITKImage::Pointer output = cast_filter->GetOutput();
    output->Update();
    output->DisconnectPipeline();
    output->SetRegions( itk_image->GetRequestedRegion() );
    field_image = ITKImage(output);

    // set zero values to very small
    const ITKImage::PixelType MINIMUM_PIXEL_VALUE = 1e-5f;
    itk::ImageRegionIterator<ImageType> iterator(field_image.getPointer(),
                                                 field_image.getPointer()->GetLargestPossibleRegion());
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
    divide_filter->SetInput1(image.getPointer());
    divide_filter->SetInput2(field_image.getPointer());
    divide_filter->Update();
    ImageType::Pointer corrected_image = divide_filter->GetOutput();
    corrected_image->DisconnectPipeline();

    return ITKImage(corrected_image);
}


void SplineInterpolationProcessor::printMetric(std::vector<ReferenceROIStatistic> rois)
{
    std::vector<ITKImage::PixelType> v;
    std::for_each (std::begin(rois), std::end(rois), [&](const ReferenceROIStatistic roi) {
        v.push_back(roi.mean_value);
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

SplineInterpolationProcessor::ReferenceROIStatistic
SplineInterpolationProcessor::calculateStatisticInROI(QVector<Point> roi, ITKImage image)
{
    ITKImage::InnerITKImage::RegionType region = image.getPointer()->GetLargestPossibleRegion();
    ITKImage::Index start = region.GetIndex();
    ITKImage::Index end = region.GetUpperIndex();

    for(Point index : roi)
    {
        if(index[0] < start[0])
            start[0] = index[0];
        if(index[1] < start[1])
            start[1] = index[1];
        if(index[2] < start[0])
            start[2] = index[2];
        if(index[0] > end[0])
            end[0] = index[0];
        if(index[1] > end[1])
            end[1] = index[1];
        if(index[2] > end[2])
            end[2] = index[2];
    }
    region.SetIndex(start);
    region.SetUpperIndex(end);

    ITKImage region_image = ExtractProcessor::process(image, region);

    typedef itk::StatisticsImageFilter<ITKImage::InnerITKImage> StatisticFilter;
    StatisticFilter::Pointer statistic_filter = StatisticFilter::New();
    statistic_filter->SetInput(region_image.getPointer());
    statistic_filter->Update();

    ReferenceROIStatistic statistic;
    statistic.point[0] = (end[0] - start[0]) / 2;
    statistic.point[1] = (end[1] - start[1]) / 2;
    statistic.point[2] = (end[2] - start[2]) / 2;
    statistic.mean_value = statistic_filter->GetMean();

    return statistic;
}
