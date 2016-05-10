#include "ITKImageProcessor.h"

#include <itkDiscreteGaussianImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkStatisticsImageFilter.h>

#include <itkImageDuplicator.h>
#include <itkImageFileWriter.h>

#include <itkBinaryThresholdImageFilter.h>
#include <itkBinaryErodeImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkImageToHistogramFilter.h>
#include <itkHistogramToLogProbabilityImageFilter.h>

#include <itkLineConstIterator.h>

#include <itkN4BiasFieldCorrectionImageFilter.h>

#include <itkShrinkImageFilter.h>
#include <itkGradientMagnitudeImageFilter.h>
#include <itkFFTWForwardFFTImageFilter.h>
#include <itkFFTWInverseFFTImageFilter.h>

#include <math.h>

const float INSIDE_MASK_VALUE = 1;
const float OUTSIDE_MASK_VALUE = 0;


#include <itkGradientImageFilter.h>
#include "retinex/ShrinkFilter.h"

#include <itkVectorIndexSelectionCastImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include <itkMaximumImageFilter.h>

#include <itkDiscreteGaussianImageFilter.h>
#include <itkComplexToRealImageFilter.h>

#include <itkBilateralImageFilter.h>
#include <itkThresholdImageFilter.h>

#include <itkMinimumMaximumImageCalculator.h>
#include <itkLaplacianImageFilter.h>

#include "itkIdentityTransform.h"
#include "itkResampleImageFilter.h"

#include <itkScaleTransform.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkExpandImageFilter.h>

#include <itkLogImageFilter.h>
#include <itkExpImageFilter.h>

#include <itkPowImageFilter.h>

ITKImageProcessor::ImageType::Pointer ITKImageProcessor::read(std::string image_file_path)
{
    itk::ImageFileReader<ImageType>::Pointer reader =
            itk::ImageFileReader<ImageType>::New();
    reader->SetFileName( image_file_path );
    try
    {
        reader->Update();
    }
    catch (itk::ExceptionObject &exception)
    {
        std::cerr << "Exception thrown while reading the image file: " <<
                     image_file_path << std::endl;
        std::cerr << exception << std::endl;
        return ImageType::New();
    }

    // rescaling is necessary for png files...
    typedef itk::RescaleIntensityImageFilter<ImageType,ImageType> RescaleFilter;
    RescaleFilter::Pointer rescale_filter = RescaleFilter::New();
    rescale_filter->SetInput(reader->GetOutput());
    rescale_filter->SetOutputMinimum(0);
    rescale_filter->SetOutputMaximum(1);
    rescale_filter->Update();

    return rescale_filter->GetOutput();
}

ITKImageProcessor::MaskImage::Pointer ITKImageProcessor::read_mask(std::string image_file_path)
{
    itk::ImageFileReader<MaskImage>::Pointer reader =
            itk::ImageFileReader<MaskImage>::New();
    reader->SetFileName( image_file_path );
    try
    {
        reader->Update();
    }
    catch (itk::ExceptionObject &exception)
    {
        std::cerr << "Exception thrown while reading the image file: " <<
                     image_file_path << std::endl;
        std::cerr << exception << std::endl;
        return MaskImage::New();
    }
    return reader->GetOutput();
}

ITKImageProcessor::ImageType::Pointer ITKImageProcessor::perform_masking(const ImageType::Pointer& image,
                            const ImageType::Pointer& mask)
{
    typedef itk::StatisticsImageFilter<ImageType> MeanValueFilter;
    MeanValueFilter::Pointer input_mean_value_filter = MeanValueFilter::New();
    input_mean_value_filter->SetInput( image );
    input_mean_value_filter->Update();
    float input_mean_value = input_mean_value_filter->GetMean();

    typedef itk::MultiplyImageFilter<ImageType, ImageType, ImageType> MultiplyFilter;
    MultiplyFilter::Pointer multiply_filter = MultiplyFilter::New();
    multiply_filter->SetInput1(image);
    multiply_filter->SetInput2(mask);
    multiply_filter->Update();
    return multiply_filter->GetOutput();
}


ITKImageProcessor::ImageType::Pointer ITKImageProcessor::bias_field(const ImageType::Pointer& image,
                            const ImageType::Pointer& mask_image)
{
    ImageType::Pointer masked = perform_masking(image, mask_image);

    typedef itk::StatisticsImageFilter<ImageType> StatisticsFilter;
    StatisticsFilter::Pointer input_mean_value_filter = StatisticsFilter::New();
    input_mean_value_filter->SetInput( masked );
    input_mean_value_filter->Update();
    float sum_intensity = input_mean_value_filter->GetSum();

    const ImageType::RegionType& region = masked->GetLargestPossibleRegion();
    const ImageType::SizeType& size = region.GetSize();

    const int voxel_count = size[0] * size[1];

    const int z = 0;

    for(int x = 0; x < size[0]; x++)
    {
        for(int y = 0; y < size[1]; y++)
        {
            ImageType::IndexType index;
            index[0] = x;
            index[1] = y;
            index[2] = z;

            itk::ImageRegionConstIterator<ImageType> mask_iterator(mask_image, region);
            float sum_of_intensities = 0;
            float sum_of_distances = 0;
            uint count_of_voxels_inside_mask = 0;
            while(! mask_iterator.IsAtEnd())
            {
                PixelType mask_intensity = mask_iterator.Get();
                if(mask_intensity != 0)
                {
                    const ImageType::IndexType mask_index = mask_iterator.GetIndex();
                    const int x_diff = x - mask_index[0];
                    const int y_diff = y - mask_index[1];
                    const int z_diff = z - mask_index[2];

                    const float distance = sqrt(x_diff*x_diff + y_diff*y_diff + z_diff*z_diff);

                    if(distance > 1e-6)
                    {
                        sum_of_distances += distance;

                        float masked_intensity = masked->GetPixel(mask_index);
                        sum_of_intensities += masked_intensity;
                        count_of_voxels_inside_mask++;
                    }
                }
                ++mask_iterator;
            }

            float bias = 0;
            itk::ImageRegionConstIterator<ImageType> mask_iterator2(mask_image, region);
            while(! mask_iterator2.IsAtEnd())
            {
                PixelType mask_intensity = mask_iterator2.Get();
                if(mask_intensity != 0)
                {
                    const ImageType::IndexType mask_index = mask_iterator2.GetIndex();
                    const int x_diff = x - mask_index[0];
                    const int y_diff = y - mask_index[1];
                    const int z_diff = z - mask_index[2];

                    const float distance = sqrt(x_diff*x_diff + y_diff*y_diff + z_diff*z_diff);
                    float masked_intensity = masked->GetPixel(mask_index);

                    if(distance > 1e-6)
                    {
                        if( abs(sum_of_distances) < 1e-6f)
                        {
                            std::cout << "?";
                            exit(1);
                        }
                        float distance_costs = distance / // average distance
                                (sum_of_distances * count_of_voxels_inside_mask);
                        float intensity_costs = masked_intensity / sum_of_intensities;

                        bias += distance_costs; // TODO: use intensity
                    }
                }
                ++mask_iterator2;
            }

            bias /= count_of_voxels_inside_mask;

            masked->SetPixel(index, bias);

            std::cout << "x=" << x << "  z=" << z <<
                        "  bias: " << bias << std::endl;
        }
    }
    return masked;
}

ITKImageProcessor::MaskImage::Pointer ITKImageProcessor::create_mask_by_maximum(
        const ImageType::Pointer& image,  const ImageType::Pointer& initial_mask)
{
    const ImageType::RegionType& region = image->GetLargestPossibleRegion();
    const ImageType::SizeType& size = region.GetSize();

    MaskImage::Pointer mask = MaskImage::New();
    mask->SetRegions(region);
    mask->Allocate();
    mask->FillBuffer(0);

    mask->SetOrigin(image->GetOrigin());
    mask->SetSpacing(image->GetSpacing());

    for(int z = 0; z < size[2]; z++)
    {
        float maximum_pixel_value = 0;
        ImageType::IndexType maximum_pixel_value_index;
        maximum_pixel_value_index[0] = 0;
        maximum_pixel_value_index[1] = 0;
        maximum_pixel_value_index[2] = 0;
        PixelType mask_inside_pixel_value = INSIDE_MASK_VALUE;
        for(int x = 0; x < size[0]; x++)
        {
            for(int y = 0; y < size[1]; y++)
            {
                ImageType::IndexType index;
                index[0] = x;
                index[1] = y;
                index[2] = z;

                mask->SetPixel(index, OUTSIDE_MASK_VALUE);

                PixelType initial_mask_pixel_value = initial_mask->GetPixel(index);
                if(initial_mask_pixel_value == OUTSIDE_MASK_VALUE)
                {
                    continue;
                }

                float pixel_value = image->GetPixel(index);
                if(pixel_value > maximum_pixel_value)
                {
                    maximum_pixel_value = pixel_value;
                    maximum_pixel_value_index = index;
                    mask_inside_pixel_value = initial_mask_pixel_value;
                }
            }
        }
        if(maximum_pixel_value_index[0] != 0 ||
                maximum_pixel_value_index[1] != 0 ||
                maximum_pixel_value_index[2] != 0)
        {
            mask->SetPixel(maximum_pixel_value_index, mask_inside_pixel_value);
        }
    }
    return mask;
}

ITKImageProcessor::ImageType::Pointer ITKImageProcessor::create_mask_by_threshold_and_erosition(
        const ImageType::Pointer& image, float threshold_pixel_value, int erosition_iterations)
{
    typedef itk::BinaryThresholdImageFilter<ImageType,ImageType> ThresholdFilter;
    ThresholdFilter::Pointer threshold_filter = ThresholdFilter::New();
    threshold_filter->SetInput(image);
    threshold_filter->SetLowerThreshold(threshold_pixel_value);
    threshold_filter->SetInsideValue(INSIDE_MASK_VALUE);
    threshold_filter->SetOutsideValue(OUTSIDE_MASK_VALUE);
    threshold_filter->Update();

    // erode...

    /*
    typedef itk::BinaryBallStructuringElement<
      ImageType::PixelType, 3> StructuringElement;
    StructuringElement erode_structure_element;
    erode_structure_element.SetRadius(3);
    erode_structure_element.CreateStructuringElement();



    typedef itk::BinaryErodeImageFilter<ImageType, ImageType, StructuringElement> ErodeFilter;
    ErodeFilter::Pointer erode_filter = ErodeFilter::New();
    erode_filter->SetInput(threshold_filter->GetOutput());
    erode_filter->SetKernel(erode_structure_element);
    erode_filter->Update();

    ImageType::Pointer mask = erode_filter->GetOutput();

    return mask;

    */
    return threshold_filter->GetOutput();
}

ITKImageProcessor::ImageType::Pointer ITKImageProcessor::create_mask(const ImageType::Pointer& image,
                                      std::vector<NodePoint> node_list)
{

    node_list.clear();
    node_list.push_back(NodePoint(50, 198, 28));
    node_list.push_back(NodePoint(62, 232, 28));
    node_list.push_back(NodePoint(73, 268, 28));
    node_list.push_back(NodePoint(298, 291, 28));
    node_list.push_back(NodePoint(311, 269, 28));
    node_list.push_back(NodePoint(328, 248, 28));
    node_list.push_back(NodePoint(90, 295, 28));

    // slice 8
    node_list.push_back(NodePoint(46, 199, 8));
    node_list.push_back(NodePoint(56, 231, 8));
    node_list.push_back(NodePoint(68, 266, 8));
    node_list.push_back(NodePoint(293, 281, 8));
    node_list.push_back(NodePoint(311, 255, 8));
    node_list.push_back(NodePoint(322, 225, 8));

    node_list.push_back(NodePoint(89, 293, 8));

    // slice 41
    node_list.push_back(NodePoint(52, 206, 41));
    node_list.push_back(NodePoint(62, 238, 41));
    node_list.push_back(NodePoint(77, 273, 41));
    node_list.push_back(NodePoint(299, 295, 41));
    node_list.push_back(NodePoint(311, 271, 41));
    node_list.push_back(NodePoint(327, 252, 41));

    node_list.push_back(NodePoint(95, 296, 41));


    // slice 12




    typedef itk::ImageDuplicator<ImageType> ImageDuplicator;
    ImageDuplicator::Pointer image_tor = ImageDuplicator::New();
    image_tor->SetInputImage(image);
    image_tor->Update();

    ImageType::Pointer mask = image_tor->GetOutput();
    const ImageType::RegionType region = mask->GetLargestPossibleRegion();


    itk::ImageRegionIterator<ImageType> image_iterator(mask, region);
    while(!image_iterator.IsAtEnd() )
    {
        image_iterator.Set(OUTSIDE_MASK_VALUE);
        ++image_iterator;
    }

    for(int i = 0; i < node_list.size(); i++)
    {
        NodePoint point = node_list[i];

        ImageType::IndexType index;
        index[0] = point.x;
        index[1] = point.y;
        index[2] = point.z;


        mask->SetPixel(index, INSIDE_MASK_VALUE);
    }

    return mask;
}



void ITKImageProcessor::write(ImageType::Pointer image, std::string file_path)
{
    // writing 32bit png
    unsigned short MAX_PIXEL_VALUE = 65535;
    typedef itk::RescaleIntensityImageFilter<ImageType, OutputImage> RescaleFilter;
    RescaleFilter::Pointer rescale_filter = RescaleFilter::New();
    rescale_filter->SetInput(image);
    rescale_filter->SetOutputMinimum(0);
    rescale_filter->SetOutputMaximum(MAX_PIXEL_VALUE);
    rescale_filter->Update();

    typedef itk::ImageFileWriter<OutputImage> WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(file_path);
    writer->SetInput(rescale_filter->GetOutput());
    writer->Update();
}


void ITKImageProcessor::write_mask(MaskImage::Pointer image, std::string file_path)
{
    typedef itk::ImageFileWriter<MaskImage> WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(file_path);
    writer->SetInput(image);
    writer->Update();
}

ITKImageProcessor::ImageType::Pointer ITKImageProcessor::histogram(const ImageType::Pointer& image)
{
    typedef itk::Statistics::ImageToHistogramFilter<ImageType> HistogramGenerator;
    HistogramGenerator::Pointer histogram_generator = HistogramGenerator::New();

    HistogramGenerator::HistogramSizeType number_of_bins(1);
    number_of_bins[0] = 255;
    histogram_generator->SetHistogramSize(number_of_bins);

    histogram_generator->SetMarginalScale(1);
    histogram_generator->SetAutoMinimumMaximum(true);

    histogram_generator->SetInput(image);
    histogram_generator->Update();

    const HistogramGenerator::HistogramType *histogram = histogram_generator->GetOutput();

    typedef itk::HistogramToLogProbabilityImageFilter<HistogramGenerator::HistogramType, ImageType> HistogramToImage;
    HistogramToImage::Pointer histogram_to_image = HistogramToImage::New();
    histogram_to_image->SetInput(histogram);
    histogram_to_image->Update();

    return histogram_to_image->GetOutput();
}

void ITKImageProcessor::histogram_data(const ImageType::Pointer& image,
                                       int bin_count,
                                       ImageType::PixelType window_from,
                                       ImageType::PixelType window_to,
                                       std::vector<double>& intensities,
                                       std::vector<double>& probabilities)
{
    typedef itk::Statistics::ImageToHistogramFilter<ImageType> HistogramGenerator;
    HistogramGenerator::Pointer histogram_generator = HistogramGenerator::New();

    HistogramGenerator::HistogramSizeType number_of_bins(1);
    number_of_bins[0] = bin_count;
    histogram_generator->SetHistogramSize(number_of_bins);

    histogram_generator->SetAutoMinimumMaximum(true);
 //   histogram_generator->SetHistogramBinMinimum(window_from);
 //   histogram_generator->SetHistogramBinMaximum(window_to);

  //  histogram_generator->SetClipBinsAtEnds(true);
    histogram_generator->SetMarginalScale(1);

    histogram_generator->SetInput(image);
    histogram_generator->Update();

    const HistogramGenerator::HistogramType *histogram = histogram_generator->GetOutput();

    /*
    const ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();
    long pixel_count = size[0] * size[1] * size[2];
    long samples_per_bin = ceil(((double)pixel_count) / bin_count); */
    float total_frequency = histogram->GetTotalFrequency();

    for(unsigned int i = 0; i < histogram->GetSize()[0]; i++)
    {
        double bin_min = histogram->GetBinMin(0, i);
        double bin_max = histogram->GetBinMax(0, i);

        if(bin_max < window_from || bin_min > window_to)
        {
            continue;
        }
        double intensity = bin_min + (bin_max - bin_min) * 0.5f;
        double probability = histogram->GetFrequency(i) / total_frequency;

        intensities.push_back(intensity);
        probabilities.push_back(probability);
    }
}

void ITKImageProcessor::find_min_max_pixel_value(const ImageType::Pointer& image,
                                            float &min_pixel_value,
                                            float &max_pixel_value)
{
    typedef itk::StatisticsImageFilter<ImageType> StatisticsFilter;
    StatisticsFilter::Pointer statistics_filter = StatisticsFilter::New();
    statistics_filter->SetInput( image );
    statistics_filter->Update();

    min_pixel_value = statistics_filter->GetMinimum();
    max_pixel_value = statistics_filter->GetMaximum();
}

void ITKImageProcessor::intensity_profile(const ImageType::Pointer & image,
                              int point1_x, int point1_y,
                              int point2_x, int point2_y,
                                     std::vector<double>& intensities,
                                     std::vector<double>& distances)
{
    ImageType::IndexType index1;
    index1[0] = point1_x;
    index1[1] = point1_y;
    index1[2] = 0;

    ImageType::IndexType index2;
    index2[0] = point2_x;
    index2[1] = point2_y;
    index2[2] = 0;

    itk::LineConstIterator<ImageType> iterator(image, index1, index2);
    while(! iterator.IsAtEnd())
    {
        PixelType intensity = iterator.Get();
        ImageType::IndexType index = iterator.GetIndex();

        const int point_x = index[0];
        const int point_y = index[1];

        const int dx = point1_x - point_x;
        const int dy = point1_y - point_y;
        const double distance = sqrt(dx*dx + dy*dy);

        intensities.push_back(intensity);
        distances.push_back(distance);

        ++iterator;
    }
}

ITKImageProcessor::ImageType::Pointer ITKImageProcessor::extract_volume(
        ImageType::Pointer image,
     unsigned int from_x, unsigned int to_x,
     unsigned int from_y, unsigned int to_y,
     unsigned int from_z, unsigned int to_z)
{
    ImageType::SizeType extract_size;
    extract_size[0] = to_x - from_x + 1;
    extract_size[1] = to_y - from_y + 1;
    bool is3D = image->GetLargestPossibleRegion().GetSize().Dimension > 2;
    if(is3D)
        extract_size[2] = to_z - from_z + 1;
    ImageType::RegionType extract_region(extract_size);

    ImageType::Pointer extracted_volume = ImageType::New();
    extracted_volume->SetRegions(extract_region);
    extracted_volume->Allocate();
    extracted_volume->SetSpacing(image->GetSpacing());

    for(unsigned int x = from_x; x <= to_x; x++)
    {
        for(unsigned int y = from_y; y <= to_y; y++)
        {
            for(unsigned int z = from_z; z <= to_z; z++)
            {
                ImageType::IndexType index_in_source;
                index_in_source[0] = x;
                index_in_source[1] = y;
                if(is3D)
                    index_in_source[2] = z;
                ImageType::PixelType pixel = image->GetPixel(index_in_source);

                ImageType::IndexType index_in_extracted;
                index_in_extracted[0] = x - from_x;
                index_in_extracted[1] = y - from_y;
                if(is3D)
                    index_in_extracted[2] = z - from_z;
                extracted_volume->SetPixel(index_in_extracted, pixel);
            }
        }
    }

    ImageType::SpacingType spacing = image->GetSpacing();
    ImageType::PointType origin = image->GetOrigin();
    origin[0] = origin[0] + from_x * spacing[0];
    origin[1] = origin[1] + from_y * spacing[1];
    if(is3D)
        origin[2] = origin[2] + from_z * spacing[2];
    extracted_volume->SetOrigin(origin);

    return extracted_volume;
}

ITKImageProcessor::ImageType::Pointer ITKImageProcessor::backward_difference_x_0_at_boundary(
        ImageType::Pointer input)
{
    ImageType::Pointer output = ImageType::New();
    ImageType::SizeType size = input->GetLargestPossibleRegion().GetSize();
    output->SetRegions(size);
    output->Allocate();

    uint first_x_index = 0;
    itk::ImageRegionIterator<ImageType> output_iterator(output,
                                                        output->GetLargestPossibleRegion());
    while(! output_iterator.IsAtEnd())
    {
        ImageType::IndexType index = output_iterator.GetIndex();

        if(index[0] == first_x_index)
        {
            ImageType::PixelType forward_difference = 0;
            output_iterator.Set(forward_difference);
        }
        else
        {
            ImageType::IndexType backward_index = index;
            backward_index[0] = backward_index[0] - 1;

            ImageType::PixelType pixel_value = input->GetPixel(index);
            ImageType::PixelType backward_pixel_value = input->GetPixel(backward_index);

            ImageType::PixelType backward_difference = pixel_value - backward_pixel_value;
            output_iterator.Set(backward_difference);

        }
        ++ output_iterator;
    }


    return output;
}

ITKImageProcessor::ImageType::Pointer ITKImageProcessor::forward_difference_x_0_at_boundary(
        ImageType::Pointer input)
{
    ImageType::Pointer output = ImageType::New();
    ImageType::SizeType size = input->GetLargestPossibleRegion().GetSize();
    output->SetRegions(size);
    output->Allocate();

    uint last_x_index = size[0] - 1;
    itk::ImageRegionIterator<ImageType> output_iterator(output,
                                                        output->GetLargestPossibleRegion());
    while(! output_iterator.IsAtEnd())
    {
        ImageType::IndexType index = output_iterator.GetIndex();

        if(index[0] == last_x_index)
        {
            ImageType::PixelType forward_difference = 0;
            output_iterator.Set(forward_difference);
        }
        else
        {
            ImageType::IndexType forward_index = index;
            forward_index[0] = forward_index[0] + 1;

            ImageType::PixelType pixel_value = input->GetPixel(index);
            ImageType::PixelType forward_pixel_value = input->GetPixel(forward_index);

            ImageType::PixelType forward_difference = forward_pixel_value - pixel_value;
            output_iterator.Set(forward_difference);

        }
        ++ output_iterator;
    }


    return output;
}


ITKImageProcessor::ImageType::Pointer ITKImageProcessor::laplace_operator(ImageType::Pointer input)
{
    ImageType::Pointer output = ImageType::New();
    ImageType::SizeType size = input->GetLargestPossibleRegion().GetSize();
    output->SetRegions(size);
    output->Allocate();

    output->SetOrigin(input->GetOrigin());
    output->SetSpacing(input->GetSpacing());

    uint last_x_index = size[0] - 1;
    uint last_y_index = size[1] - 1;
    itk::ImageRegionIterator<ImageType> output_iterator(output,
                                                        output->GetLargestPossibleRegion());
    while(! output_iterator.IsAtEnd())
    {
        ImageType::IndexType index = output_iterator.GetIndex();

        ImageType::PixelType laplace_value = 0;
        ImageType::PixelType pixel_value = input->GetPixel(index);

        if(index[0] > 0)
        {
            ImageType::IndexType x_backward_index = index;
            x_backward_index[0] = x_backward_index[0] - 1;
            ImageType::PixelType x_backward_value = input->GetPixel(x_backward_index);
            laplace_value += (x_backward_value - pixel_value);
        }
        if(index[0] < last_x_index)
        {
            ImageType::IndexType x_forward_index = index;
            x_forward_index[0] = x_forward_index[0] + 1;
            ImageType::PixelType x_forward_value = input->GetPixel(x_forward_index);
            laplace_value += (x_forward_value - pixel_value);
        }

        if(index[1] > 0)
        {
            ImageType::IndexType y_backward_index = index;
            y_backward_index[1] = y_backward_index[1] - 1;
            ImageType::PixelType y_backward_value = input->GetPixel(y_backward_index);
            laplace_value += (y_backward_value - pixel_value);
        }
        if(index[1] < last_y_index)
        {
            ImageType::IndexType y_forward_index = index;
            y_forward_index[1] = y_forward_index[1] + 1;
            ImageType::PixelType y_forward_value = input->GetPixel(y_forward_index);
            laplace_value += (y_forward_value - pixel_value);
        }

        output_iterator.Set(laplace_value);
        ++ output_iterator;
    }


    return output;
}

ITKImageProcessor::ImageType::Pointer
ITKImageProcessor::laplace_operator_projected(ImageType::Pointer input)
{
    ImageType::SizeType size = input->GetLargestPossibleRegion().GetSize();

    ImageType::Pointer grad_x = ImageType::New();
    grad_x->SetRegions(size);
    grad_x->Allocate();

    ImageType::Pointer grad_y = ImageType::New();
    grad_y->SetRegions(size);
    grad_y->Allocate();

    uint last_x_index = size[0] - 1;
    uint last_y_index = size[1] - 1;
    itk::ImageRegionIterator<ImageType> grad_x_iterator(grad_x,
              grad_x->GetLargestPossibleRegion());
    itk::ImageRegionIterator<ImageType> grad_y_iterator(grad_y,
              grad_y->GetLargestPossibleRegion());
    // gradients
    while(! grad_x_iterator.IsAtEnd())
    {
        ImageType::IndexType index = grad_x_iterator.GetIndex();
        ImageType::PixelType pixel_value = input->GetPixel(index);

        ImageType::PixelType grad_x_value = 0;
        ImageType::PixelType grad_y_value = 0;

        if(index[0] < last_x_index)
        {
            ImageType::IndexType x_forward_index = index;
            x_forward_index[0] = x_forward_index[0] + 1;
            ImageType::PixelType x_forward_value = input->GetPixel(x_forward_index);
            grad_x_value = x_forward_value - pixel_value;
        }
        else
        {
            grad_x_value = 0;
        }

        if(index[1] < last_y_index)
        {
            ImageType::IndexType y_forward_index = index;
            y_forward_index[1] = y_forward_index[1] + 1;
            ImageType::PixelType y_forward_value = input->GetPixel(y_forward_index);
            grad_y_value = y_forward_value - pixel_value;
        }
        else
        {
            grad_y_value = 0;
        }

        float grad_magnitude = sqrt(grad_x_value*grad_x_value +
                                    grad_y_value*grad_y_value);
        if(grad_magnitude < 1)
        {
            grad_magnitude = 1;
        }

        grad_x_iterator.Set(grad_x_value / grad_magnitude);
        grad_y_iterator.Set(grad_y_value / grad_magnitude);

        ++ grad_x_iterator;
        ++ grad_y_iterator;
    }

    itk::ImageRegionConstIterator<ImageType> grad_x_const_iterator(grad_x,
              grad_x->GetLargestPossibleRegion());
    itk::ImageRegionConstIterator<ImageType> grad_y_const_iterator(grad_y,
              grad_y->GetLargestPossibleRegion());

    ImageType::Pointer output = ImageType::New();
    output->SetRegions(size);
    output->Allocate();

    output->SetOrigin(input->GetOrigin());
    output->SetSpacing(input->GetSpacing());

    itk::ImageRegionIterator<ImageType> output_iterator(output,
              output->GetLargestPossibleRegion());

    // divergence
    while(!output_iterator.IsAtEnd())
    {
        ImageType::IndexType index = output_iterator.GetIndex();
        ImageType::PixelType grad_x_pixel = grad_x_const_iterator.Get();
        ImageType::PixelType grad_y_pixel = grad_y_const_iterator.Get();
        ImageType::PixelType div_x_pixel = 0;
        ImageType::PixelType div_y_pixel = 0;

        if(index[0] > 0)
        {
            ImageType::IndexType x_backward_index = index;
            x_backward_index[0] = x_backward_index[0] - 1;
            ImageType::PixelType x_backward_value = grad_x->GetPixel(x_backward_index);
            div_x_pixel = x_backward_value - grad_x_pixel;
        } // else div_x keeps the value 0 (neumann boundary conditions)

        if(index[1] > 0)
        {
            ImageType::IndexType y_backward_index = index;
            y_backward_index[1] = y_backward_index[1] - 1;
            ImageType::PixelType y_backward_value = grad_y->GetPixel(y_backward_index);
            div_y_pixel = y_backward_value - grad_y_pixel;
        } // else div_y keeps the value 0 (neumann boundary conditions)

        output_iterator.Set(div_x_pixel + div_y_pixel);

        ++ output_iterator;
        ++ grad_x_const_iterator;
        ++ grad_y_const_iterator;
    }


    return output;
}

void ITKImageProcessor::gradients_projected_and_laplace(
     ImageType::Pointer input,
     ImageType::Pointer& gradient_x_projected,
     ImageType::Pointer& gradient_y_projected,
     ImageType::Pointer& laplace_image)
{
    ImageType::SizeType size = input->GetLargestPossibleRegion().GetSize();

    gradient_x_projected = ImageType::New();
    gradient_x_projected->SetRegions(size);
    gradient_x_projected->Allocate();
    gradient_x_projected->SetOrigin(input->GetOrigin());
    gradient_x_projected->SetSpacing(input->GetSpacing());

    gradient_y_projected = ImageType::New();
    gradient_y_projected->SetRegions(size);
    gradient_y_projected->Allocate();
    gradient_y_projected->SetOrigin(input->GetOrigin());
    gradient_y_projected->SetSpacing(input->GetSpacing());

    uint last_x_index = size[0] - 1;
    uint last_y_index = size[1] - 1;
    itk::ImageRegionIterator<ImageType> gradient_x_iterator(gradient_x_projected,
              gradient_x_projected->GetLargestPossibleRegion());
    itk::ImageRegionIterator<ImageType> gradient_y_iterator(gradient_y_projected,
              gradient_y_projected->GetLargestPossibleRegion());
    // gradients
    while(! gradient_x_iterator.IsAtEnd())
    {
        ImageType::IndexType index = gradient_x_iterator.GetIndex();
        ImageType::PixelType pixel_value = input->GetPixel(index);

        ImageType::PixelType gradient_x_value = 0;
        ImageType::PixelType gradient_y_value = 0;

        if(index[0] < last_x_index)
        {
            ImageType::IndexType x_forward_index = index;
            x_forward_index[0] = x_forward_index[0] + 1;
            ImageType::PixelType x_forward_value = input->GetPixel(x_forward_index);
            gradient_x_value = x_forward_value - pixel_value;
        }
        else
        {
            gradient_x_value = 0;
        }

        if(index[1] < last_y_index)
        {
            ImageType::IndexType y_forward_index = index;
            y_forward_index[1] = y_forward_index[1] + 1;
            ImageType::PixelType y_forward_value = input->GetPixel(y_forward_index);
            gradient_y_value = y_forward_value - pixel_value;
        }
        else
        {
            gradient_y_value = 0;
        }

        float grad_magnitude = sqrt(gradient_x_value*gradient_x_value +
                                    gradient_y_value*gradient_y_value);
        if(grad_magnitude < 1)
        {
            grad_magnitude = 1;
        }

        gradient_x_iterator.Set(gradient_x_value / grad_magnitude);
        gradient_y_iterator.Set(gradient_y_value / grad_magnitude);

        ++ gradient_x_iterator;
        ++ gradient_y_iterator;
    }

    itk::ImageRegionConstIterator<ImageType> gradient_x_const_iterator(gradient_x_projected,
              gradient_x_projected->GetLargestPossibleRegion());
    itk::ImageRegionConstIterator<ImageType> gradient_y_const_iterator(gradient_y_projected,
              gradient_y_projected->GetLargestPossibleRegion());

    laplace_image = ImageType::New();
    laplace_image->SetRegions(size);
    laplace_image->Allocate();

    laplace_image->SetOrigin(input->GetOrigin());
    laplace_image->SetSpacing(input->GetSpacing());

    itk::ImageRegionIterator<ImageType> laplace_iterator(laplace_image,
              laplace_image->GetLargestPossibleRegion());

    // divergence
    while(!laplace_iterator.IsAtEnd())
    {
        ImageType::IndexType index = laplace_iterator.GetIndex();
        ImageType::PixelType gradient_x_pixel = gradient_x_const_iterator.Get();
        ImageType::PixelType gradient_y_pixel = gradient_y_const_iterator.Get();
        ImageType::PixelType div_x_pixel = 0;
        ImageType::PixelType div_y_pixel = 0;

        if(index[0] > 0)
        {
            ImageType::IndexType x_backward_index = index;
            x_backward_index[0] = x_backward_index[0] - 1;
            ImageType::PixelType x_backward_value = gradient_x_projected->GetPixel(x_backward_index);
            div_x_pixel = x_backward_value - gradient_x_pixel;
        } // else div_x keeps the value 0 (neumann boundary conditions)

        if(index[1] > 0)
        {
            ImageType::IndexType y_backward_index = index;
            y_backward_index[1] = y_backward_index[1] - 1;
            ImageType::PixelType y_backward_value = gradient_y_projected->GetPixel(y_backward_index);
            div_y_pixel = y_backward_value - gradient_y_pixel;
        } // else div_y keeps the value 0 (neumann boundary conditions)

        laplace_iterator.Set(div_x_pixel + div_y_pixel);

        ++ laplace_iterator;
        ++ gradient_x_const_iterator;
        ++ gradient_y_const_iterator;
    }
}

void ITKImageProcessor::removeSensorSensitivity_FFTOperators(const ImageType::SizeType size,
             ImageType::Pointer& fft_laplace,
             ComplexImageType::Pointer& fft_gradient_h,
             ComplexImageType::Pointer& fft_gradient_v)
{
    /* MATLAB CODE:
    k_x = (0:(width-1)) ./ width;
    k_y = ((0:(height-1)) ./ height)';
    fft_laplace = 4 - 2 * (repmat(cos(k_x),height,1) + repmat(cos(k_y),1,width));
    fft_gradient_h = repmat(1 - exp(-i*k_x),height,1);
    fft_gradient_v = repmat(1 - exp(-i*k_y),1,width);
    */
    fft_laplace = ImageType::New(); fft_laplace->SetRegions(size); fft_laplace->Allocate();
    fft_gradient_h = ComplexImageType::New(); fft_gradient_h->SetRegions(size); fft_gradient_h->Allocate();
    fft_gradient_v = ComplexImageType::New(); fft_gradient_v->SetRegions(size); fft_gradient_v->Allocate();

    const ComplexPixelType i(0,1);

    const int width = size[0];
    const int height = size[1];
    for(int y = 0; y < height; y++)
    {
        const float k_y = y / ((float) height);

        const ImageType::PixelType laplace_value_y = cos(k_y);
        const ComplexImageType::PixelType fft_gradient_v_value = 1.0f - std::exp(-i * k_y);
        for(int x = 0; x < width; x++)
        {
            ImageType::IndexType index; index[0] = x; index[1] = y;
            const float k_x = x / ((float) width);

            fft_laplace->SetPixel(index, 4 - 2 * (cos(k_x) + laplace_value_y));
            fft_gradient_v->SetPixel(index, fft_gradient_v_value);
            fft_gradient_h->SetPixel(index, 1.0f - std::exp(-i * k_x));
        }
    }
}

template<typename T>
typename T::Pointer ITKImageProcessor::clone(const typename T::Pointer image)
{
    typedef itk::ImageDuplicator< T > DuplicatorType;
    typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(image);
    duplicator->Update();

    typename T::Pointer clone = duplicator->GetOutput();
    clone->SetSpacing(image->GetSpacing());
    clone->SetOrigin(image->GetOrigin());
    return clone;
}

typename ITKImageProcessor::ImageType::Pointer ITKImageProcessor::cloneImage(const typename ImageType::Pointer image)
{
    if(image.IsNull())
        return nullptr;

    typedef ImageType T;
    typedef itk::ImageDuplicator< T > DuplicatorType;
    typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(image);
    duplicator->Update();

    typename T::Pointer clone = duplicator->GetOutput();
    clone->SetSpacing(image->GetSpacing());
    clone->SetOrigin(image->GetOrigin());
    return clone;
}


template<typename T2>
typename T2::PixelType ITKImageProcessor::SumAllPixels(const typename T2::Pointer image)
{
    typename T2::PixelType sum = 0;
    typedef itk::ImageRegionConstIterator<T2> Iterator;
    Iterator iterator(image, image->GetLargestPossibleRegion());
    while(!iterator.IsAtEnd())
    {
        sum += iterator.Value();
        ++iterator;
    }
    return sum;
}

ITKImageProcessor::ImageType::Pointer ITKImageProcessor::gammaEnhancement(
        ImageType::Pointer S1,
        ImageType::Pointer L1, const float gamma)
{
    ImageType::Pointer S = clone<ImageType>(S1);
    ImageType::Pointer L = clone<ImageType>(L1);

    typedef itk::RescaleIntensityImageFilter<ImageType> RescaleIntensityFilter;
    RescaleIntensityFilter::Pointer rescale_intensity_filter = RescaleIntensityFilter::New();
    rescale_intensity_filter->SetInput(S);
    rescale_intensity_filter->SetOutputMinimum(0);
    rescale_intensity_filter->SetOutputMaximum(1);
    rescale_intensity_filter->Update();

    typedef itk::DivideImageFilter<ImageType, ImageType, ImageType> DivideFilter;
    /*
    DivideFilter::Pointer divide_filter2 = DivideFilter::New();
    divide_filter2->SetInput1(L);
    divide_filter2->SetConstant2(255); // white
    divide_filter2->Update();

    RescaleIntensityFilter::Pointer rescale_intensity_filter2 = RescaleIntensityFilter::New();
    rescale_intensity_filter2->SetInput(L);
    rescale_intensity_filter2->SetOutputMinimum(0.001);
    rescale_intensity_filter2->SetOutputMaximum(1);
    rescale_intensity_filter2->Update();
    */

    typedef itk::PowImageFilter<ImageType> PowFilter;
    PowFilter::Pointer pow_filter = PowFilter::New();
    pow_filter->SetInput1(L);
    pow_filter->SetConstant2(1 - 1 / gamma);
    pow_filter->Update();

    DivideFilter::Pointer divide_filter = DivideFilter::New();
    divide_filter->SetInput1(rescale_intensity_filter->GetOutput());
    divide_filter->SetInput2(pow_filter->GetOutput());
    divide_filter->Update();

    return divide_filter->GetOutput();
}

void ITKImageProcessor::multiScaleRetinex(
        ImageType::Pointer image,
        std::vector<MultiScaleRetinex::Scale*> scales,
        std::function<void(ImageType::Pointer)> finished_callback)
{
    //normalize weights
    float weight_sum = 0;
    for(MultiScaleRetinex::Scale* scale : scales)
        weight_sum += scale->weight;

    ImageType::Pointer input_image = clone<ImageType>(image);
    ImageType::RegionType region = input_image->GetLargestPossibleRegion();

    ImageType::Pointer reflectance = ImageType::New();
    reflectance->SetRegions(region);
    reflectance->Allocate();
    reflectance->FillBuffer(0);

    typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType> GaussFilter;
    typedef itk::LogImageFilter<ImageType, ImageType> LogFilter;
    typedef itk::SubtractImageFilter<ImageType, ImageType, ImageType> SubtractFilter;
    typedef itk::AddImageFilter<ImageType, ImageType, ImageType> AddFilter;
    typedef itk::MultiplyImageFilter<ImageType, ImageType, ImageType> MultiplyFilter;

    AddFilter::Pointer add_filter_input = AddFilter::New();
    add_filter_input->SetInput1(input_image);
    add_filter_input->SetConstant2(1);
    add_filter_input->Update();

    LogFilter::Pointer log_filter_input = LogFilter::New();
    log_filter_input->SetInput(add_filter_input->GetOutput());
    log_filter_input->Update();

    for(MultiScaleRetinex::Scale* scale : scales)
    {
        GaussFilter::Pointer gauss_filter = GaussFilter::New();
        gauss_filter->SetInput(input_image);
        gauss_filter->SetMaximumKernelWidth(32);
        gauss_filter->SetVariance(scale->sigma);
        gauss_filter->Update();

        AddFilter::Pointer add_filter1 = AddFilter::New();
        add_filter1->SetInput1(gauss_filter->GetOutput());
        add_filter1->SetConstant2(1);
        add_filter1->Update();

        LogFilter::Pointer log_filter1 = LogFilter::New();
        log_filter1->SetInput(add_filter1->GetOutput());
        log_filter1->Update();

        SubtractFilter::Pointer subtract_filter = SubtractFilter::New();
        subtract_filter->SetInput1(log_filter_input->GetOutput());
        subtract_filter->SetInput2(log_filter1->GetOutput());
        subtract_filter->Update();

        MultiplyFilter::Pointer multiply_filter = MultiplyFilter::New();
        multiply_filter->SetInput1(subtract_filter->GetOutput());
        multiply_filter->SetConstant2(scale->weight / weight_sum);
        multiply_filter->Update();

        AddFilter::Pointer add_filter = AddFilter::New();
        add_filter->SetInput1(reflectance);
        add_filter->SetInput2(multiply_filter->GetOutput());
        add_filter->Update();
        reflectance = add_filter->GetOutput();

    }

    finished_callback(reflectance);
}

void ITKImageProcessor::removeSensorSensitivity(
        ImageType::Pointer input_image0,
        const float alpha,
        const float beta,
        const int pyramid_levels,
        const int iteration_count_factor,
        const bool with_max_contraint,
        std::function<void(ImageType::Pointer,uint,uint)> iteration_callback,
        std::function<void(ImageType::Pointer,ImageType::Pointer)> finished_callback)
{
    // according to Kimmel 2003
    ImageType::Pointer input_image1 = clone<ImageType>(input_image0);

    typedef itk::RescaleIntensityImageFilter<ImageType> RescaleIntensityFilter;
    RescaleIntensityFilter::Pointer rescale_intensity_filter = RescaleIntensityFilter::New();
    rescale_intensity_filter->SetInput(input_image1);
    rescale_intensity_filter->SetOutputMinimum(0);
    rescale_intensity_filter->SetOutputMaximum(1);
    rescale_intensity_filter->Update();

    typedef itk::AddImageFilter<ImageType, ImageType> AddFilter;
    AddFilter::Pointer add_filter1 = AddFilter::New();
    add_filter1->SetInput1(rescale_intensity_filter->GetOutput());
    add_filter1->SetConstant2(1);
    add_filter1->Update();

    typedef itk::LogImageFilter<ImageType, ImageType> LogFilter;
    LogFilter::Pointer log_filter = LogFilter::New();
    log_filter->SetInput(add_filter1->GetOutput());
    log_filter->Update();

    ImageType::Pointer input_image = log_filter->GetOutput();

    // INIT gauss pyramid
    typedef itk::ShrinkImageFilter<ImageType, ImageType> ShrinkImageFilter;
    typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType> GaussFilter;
    std::vector<ImageType::Pointer> S;
    std::vector<ImageType::Pointer> L;
    for(int pyramid_level = 0; pyramid_level < pyramid_levels; pyramid_level++)
    {
        GaussFilter::Pointer gauss_filter = GaussFilter::New();
        gauss_filter->SetVariance(4);
        gauss_filter->SetMaximumKernelWidth(16);
        if(pyramid_level == 0)
            gauss_filter->SetInput(input_image);
        else
            gauss_filter->SetInput(S[pyramid_level-1]);
        gauss_filter->Update();

        if(pyramid_level > 0)
        {
            // downsscaling
            ShrinkImageFilter::Pointer shrink_filter =  ShrinkImageFilter::New();
            shrink_filter->SetInput(gauss_filter->GetOutput());
            shrink_filter->SetShrinkFactors(2);
            shrink_filter->Update();
            S.push_back( shrink_filter->GetOutput() ) ;
        }
        else
            S.push_back( gauss_filter->GetOutput() );

        L.push_back(nullptr); // will be filled later from end to start
    }

    // INIT filter types
    typedef itk::LaplacianImageFilter<ImageType, ImageType> LaplacianFilter;
    typedef itk::SubtractImageFilter<ImageType, ImageType> SubtractFilter;
    typedef itk::MultiplyImageFilter<ImageType, ImageType> MultiplyFilter;
    typedef itk::MaximumImageFilter<ImageType, ImageType> MaximumFilter;

    // Reverse loop pyramid

    ImageType::SpacingType spacing;
    spacing.Fill(1);

    for(int pyramid_level = pyramid_levels - 1; pyramid_level >= 0; pyramid_level--)
    {
        ImageType::Pointer Sc = S[pyramid_level]; // S of current level
        Sc->SetOrigin(input_image->GetOrigin());
        Sc->SetSpacing(spacing);
        ImageType::Pointer Lp = nullptr; // L of the previous level
        if(pyramid_level < pyramid_levels - 1)
        {
            // upscale the previous solution of the illumination
            Lp = L[pyramid_level + 1];

            typedef itk::ExpandImageFilter<ImageType, ImageType> ExpandFilter;
            ExpandFilter::Pointer expand_filter = ExpandFilter::New();
            expand_filter->SetInput(Lp);
            expand_filter->SetExpandFactors(2);
            expand_filter->Update();

            Lp = expand_filter->GetOutput();
            Lp->SetOrigin(Sc->GetOrigin());
            Lp->SetSpacing(Sc->GetSpacing());
        }
        else
        {
            // take the maximum value of the input image as initial condition
            const ImageType::RegionType region = Sc->GetLargestPossibleRegion();
            Lp = ImageType::New();
            Lp->SetRegions(region);
            Lp->Allocate();
            typedef itk::MinimumMaximumImageCalculator<ImageType> MaxCalculator;
            MaxCalculator::Pointer max_calculator = MaxCalculator::New();
            max_calculator->SetImage(Sc);
            max_calculator->ComputeMaximum();
            ImageType::PixelType max_value = max_calculator->GetMaximum();

            Lp->FillBuffer(max_value);
        }

        const int iteration_count = iteration_count_factor * (pyramid_level + 1);
        for(int j = 0; j < iteration_count; j++)
        {
            LaplacianFilter::Pointer laplacian_filterS = LaplacianFilter::New();
            laplacian_filterS->SetInput(Sc);
            laplacian_filterS->Update();

            LaplacianFilter::Pointer laplacian_filterL = LaplacianFilter::New();
            laplacian_filterL->SetInput(Lp);
            laplacian_filterL->Update();

            SubtractFilter::Pointer subtract_filter = SubtractFilter::New();
            subtract_filter->SetInput1(Lp);
            subtract_filter->SetInput2(Sc);
            subtract_filter->Update(); // error

            MultiplyFilter::Pointer multiply_filter = MultiplyFilter::New();
            multiply_filter->SetInput1(subtract_filter->GetOutput());
            multiply_filter->SetConstant2(alpha);
            multiply_filter->Update();

            SubtractFilter::Pointer subtract_filter2 = SubtractFilter::New();
            subtract_filter2->SetInput1(laplacian_filterL->GetOutput());
            subtract_filter2->SetInput2(laplacian_filterS->GetOutput());
            subtract_filter2->Update();

            MultiplyFilter::Pointer multiply_filter2 = MultiplyFilter::New();
            multiply_filter2->SetInput1(subtract_filter2->GetOutput());
            multiply_filter2->SetConstant2(beta);
            multiply_filter2->Update();

            SubtractFilter::Pointer subtract_filter3 = SubtractFilter::New();
            subtract_filter3->SetInput1(multiply_filter->GetOutput());
            subtract_filter3->SetInput2(multiply_filter2->GetOutput());
            subtract_filter3->Update();

            AddFilter::Pointer add_filter = AddFilter::New();
            add_filter->SetInput1(laplacian_filterL->GetOutput());
            add_filter->SetInput2(subtract_filter3->GetOutput());
            add_filter->Update(); // G

            // calculating mu_nsd
            MultiplyFilter::Pointer multiply_filter3 = MultiplyFilter::New();
            multiply_filter3->SetInput1(add_filter->GetOutput());
            multiply_filter3->SetInput2(add_filter->GetOutput());
            multiply_filter3->Update();
            const ImageType::PixelType mu_a = SumAllPixels<ImageType>(multiply_filter3->GetOutput());

            LaplacianFilter::Pointer laplacian_filterG = LaplacianFilter::New();
            laplacian_filterG->SetInput(add_filter->GetOutput());
            laplacian_filterG->Update();

            MultiplyFilter::Pointer multiply_filter4 = MultiplyFilter::New();
            multiply_filter4->SetInput1(add_filter->GetOutput());
            multiply_filter4->SetInput2(laplacian_filterG->GetOutput());
            multiply_filter4->Update();
            const ImageType::PixelType mu_b = SumAllPixels<ImageType>(multiply_filter4->GetOutput());


            const ImageType::PixelType mu_nsd = mu_a / (alpha * mu_a + (1 + beta) * mu_b);

            MultiplyFilter::Pointer multiply_filterStep = MultiplyFilter::New();
            multiply_filterStep->SetInput1(add_filter->GetOutput());
            multiply_filterStep->SetConstant2(mu_nsd);
            multiply_filterStep->Update();

            SubtractFilter::Pointer subtract_filterStep = SubtractFilter::New();
            subtract_filterStep->SetInput1(Lp);
            subtract_filterStep->SetInput2(multiply_filterStep->GetOutput());
            subtract_filterStep->Update();

            if(with_max_contraint)
            {
                // project to constraint L > S
                MaximumFilter::Pointer maximum_filter = MaximumFilter::New();
                maximum_filter->SetInput1(subtract_filterStep->GetOutput());
                maximum_filter->SetInput2(Sc);
                maximum_filter->Update();
                L[pyramid_level] = maximum_filter->GetOutput();
            }
            else
                L[pyramid_level] = subtract_filterStep->GetOutput();

            Lp = L[pyramid_level];

            if(iteration_callback != nullptr)
                iteration_callback(Lp, j, iteration_count);
        }

    }


    /*
    typedef itk::DivideImageFilter<ImageType, ImageType, ImageType> DivideFilter;
    DivideFilter::Pointer divide_filter = DivideFilter::New();
    divide_filter->SetInput1(input_image);
    divide_filter->SetInput2(output_illumination);
    divide_filter->Update();
    */

    SubtractFilter::Pointer subtract_filterR = SubtractFilter::New();
    subtract_filterR->SetInput1(input_image);
    subtract_filterR->SetInput2(L[0]);
    subtract_filterR->Update();

    typedef itk::ExpImageFilter<ImageType, ImageType> ExpFilter;
    ExpFilter::Pointer exp_filterR = ExpFilter::New();
    exp_filterR->SetInput(subtract_filterR->GetOutput());
    exp_filterR->Update();

    SubtractFilter::Pointer subtract_filterR2 = SubtractFilter::New();
    subtract_filterR2->SetInput1(exp_filterR->GetOutput());
    subtract_filterR2->SetConstant2(1);
    subtract_filterR2->Update();

    ImageType::Pointer output_reflectance = subtract_filterR2->GetOutput();
    output_reflectance->SetOrigin(input_image->GetOrigin());
    output_reflectance->SetSpacing(input_image->GetSpacing());

    ExpFilter::Pointer exp_filterL = ExpFilter::New();
    exp_filterL->SetInput(L[0]);
    exp_filterL->Update();

    SubtractFilter::Pointer subtract_filterL2 = SubtractFilter::New();
    subtract_filterL2->SetInput1(exp_filterL->GetOutput());
    subtract_filterL2->SetConstant2(1);
    subtract_filterL2->Update();

    ImageType::Pointer output_illumination = subtract_filterL2->GetOutput();
    output_illumination->SetOrigin(input_image->GetOrigin());
    output_illumination->SetSpacing(input_image->GetSpacing());


    finished_callback(output_reflectance,output_illumination);

}


ITKImageProcessor::ImageType::Pointer ITKImageProcessor::gradient_magnitude_image(ImageType::Pointer input)
{
    typedef itk::GradientMagnitudeImageFilter<ImageType,ImageType> Filter;
    Filter::Pointer filter = Filter::New();
    filter->SetInput(input);
    filter->Update();
    return filter->GetOutput();
}


ITKImageProcessor::ImageType::Pointer ITKImageProcessor::bilateralFilter(ImageType::Pointer image,
                                          float sigma_spatial_distance,
                                          float sigma_intensity_distance,
                                          int kernel_size)
{
    typedef itk::BilateralImageFilter<ImageType,ImageType> BilateralFilter;
    BilateralFilter::Pointer filter = BilateralFilter::New();
    filter->SetInput(image);
    filter->SetDomainSigma(sigma_intensity_distance);
    filter->SetRangeSigma(sigma_spatial_distance);
    filter->SetAutomaticKernelSize(false);
    BilateralFilter::SizeType radius;
    radius.Fill(kernel_size);
    filter->SetRadius(radius);

    filter->Update();
    return filter->GetOutput();

    /*
     *default values:
          this->m_Radius.Fill(1);
          this->m_AutomaticKernelSize = true;
          this->m_DomainSigma.Fill(4.0);
          this->m_RangeSigma = 50.0;
          this->m_FilterDimensionality = ImageDimension;
          this->m_NumberOfRangeGaussianSamples = 100;
          this->m_DynamicRange = 0.0;
          this->m_DynamicRangeUsed = 0.0;
          this->m_DomainMu = 2.5;  // keep small to keep kernels small
          this->m_RangeMu = 4.0;
    */
}

ITKImageProcessor::ImageType::Pointer ITKImageProcessor::threshold(
        ImageType::Pointer image,
        ImageType::PixelType lower_threshold_value,
        ImageType::PixelType upper_threshold_value,
        ImageType::PixelType outside_pixel_value)
{

    typedef itk::ThresholdImageFilter<ImageType> ThresholdImageFilter;
    ThresholdImageFilter::Pointer filter = ThresholdImageFilter::New();
    filter->SetInput(image);
    filter->ThresholdOutside(lower_threshold_value, upper_threshold_value);
    filter->SetOutsideValue(outside_pixel_value);
    filter->Update();

    return filter->GetOutput();
}

ITKImageProcessor::ImageType::Pointer ITKImageProcessor::splineFit(
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
