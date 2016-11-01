#include "BSplineInterpolationProcessor.h"

#include <itkPointSet.h>
#include <itkBSplineScatteredDataPointSetToImageFilter.h>
#include <itkVectorIndexSelectionCastImageFilter.h>

BSplineInterpolationProcessor::BSplineInterpolationProcessor()
{

}

ITKImage BSplineInterpolationProcessor::process(
        ITKImage image, ITKImage mask,
        uint spline_order, uint number_of_fitting_levels)
{
    return image.depth == 1 ?
                processDimensions<2>(image, mask, spline_order, number_of_fitting_levels) :
                processDimensions<3>(image, mask, spline_order, number_of_fitting_levels);

}

template<unsigned int NDimension>
ITKImage BSplineInterpolationProcessor::processDimensions(
        ITKImage image, ITKImage mask,
        uint spline_order, uint number_of_fitting_levels) {

    typedef itk::Vector<ITKImage::PixelType, 1> ScalarType;
    typedef itk::PointSet<ScalarType, NDimension> PointSet;
    typename PointSet::Pointer field_points = PointSet::New();
    field_points->Initialize();

    typedef itk::Image<ScalarType, NDimension> ScalarImageType;
    typedef typename itk::BSplineScatteredDataPointSetToImageFilter<PointSet, ScalarImageType>
            BSplineFilter;
    typename BSplineFilter::WeightsContainerType::Pointer weights = BSplineFilter::WeightsContainerType::New();
    weights->Initialize();

    int scalar_index = 0;
    typedef ITKImage::InnerITKImage ImageType;
    ImageType::Pointer itk_image = image.getPointer();
    image.foreachPixel([&mask, &itk_image, &weights, &field_points, &scalar_index]
                       (uint x, uint y, uint z, ITKImage::PixelType pixel_value) {
        if(!mask.isNull() && mask.getPixel(x, y, z) == 0)
            return;

        typename ImageType::IndexType index;
        index[0] = x;
        index[1] = y;

        if(NDimension > 2)
            index[2] = z;

        ScalarType scalar;
        scalar[0] = pixel_value;

        ImageType::PointType point;
        itk_image->TransformIndexToPhysicalPoint(index, point);
        typename  PointSet::PointType setpoint;
        setpoint[0] = point[0]; setpoint[1] = point[1];
        if(NDimension > 2) {
            setpoint[2] = point[2];
        }

        field_points->SetPointData(scalar_index, scalar);
        field_points->SetPoint(scalar_index, setpoint);
        weights->InsertElement(scalar_index, 1.0);

        scalar_index++;
    });

    typename BSplineFilter::Pointer bspliner = BSplineFilter::New();
    typename BSplineFilter::ArrayType numberOfControlPointsArray;
    typename BSplineFilter::ArrayType numberOfFittingLevelsArray;
    numberOfControlPointsArray.Fill(spline_order + 1);
    numberOfFittingLevelsArray.Fill(number_of_fitting_levels);

    auto image_origin = itk_image->GetOrigin();
    ImageType::SizeType image_size = itk_image->GetLargestPossibleRegion().GetSize();
    auto image_spacing = itk_image->GetSpacing();
 //    auto image_direction = itk_image->GetSpacing();

    typename BSplineFilter::PointType origin;
    origin[0] = image_origin[0]; origin[1] = image_origin[1];

    typename BSplineFilter::SpacingType spacing;
    spacing[0] = image_spacing[0]; spacing[1] = image_spacing[1];
    typename BSplineFilter::SizeType size;
    size[0] = image_size[0]; size[1] = image_size[1];
  //   typename BSplineFilter::DirectionType direction;
  //  direction[0] = image_direction[0]; direction[1] = image_direction[1];
    if(NDimension > 2) {
        spacing[2] = image_spacing[2];
        size[2] = image_size[2];
        origin[2] = image_origin[2];
  //      direction[2] = image_direction[2];
    }
    bspliner->SetSpacing(spacing);
    bspliner->SetSize(size);
 //   bspliner->SetDirection(itk_image->GetDirection());
    bspliner->SetOrigin(origin);

    bspliner->SetGenerateOutputImage(true);

    bspliner->SetNumberOfLevels(numberOfFittingLevelsArray);
    bspliner->SetSplineOrder(spline_order);
    bspliner->SetNumberOfControlPoints(numberOfControlPointsArray);

    bspliner->SetInput(field_points);
    bspliner->SetPointWeights(weights);

    try {
        bspliner->Update();
    }
    catch(itk::ExceptionObject exception) {
        exception.Print(std::cerr);
        throw exception;
    }

    typedef itk::VectorIndexSelectionCastImageFilter<ScalarImageType, ImageType> CastFilter;
    typename CastFilter::Pointer cast_filter = CastFilter::New();
    cast_filter->SetInput(bspliner->GetOutput());
    cast_filter->SetIndex(0); // real component of the complex image

    ImageType::Pointer output = cast_filter->GetOutput();
    output->Update();
    output->DisconnectPipeline();
    output->SetRegions(itk_image->GetRequestedRegion());

    return ITKImage(output);
}

