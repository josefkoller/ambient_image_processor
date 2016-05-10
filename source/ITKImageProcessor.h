#ifndef ITKIMAGEPROCESSOR_H
#define ITKIMAGEPROCESSOR_H

#include <itkImage.h>
#include <itkImageFileReader.h>

#include <functional>
#include <vector>
#include <complex>

#include "retinex/MultiScaleRetinex.h"

const unsigned int InputDimension = 2;

class ITKImageProcessor
{
public:
    // MHA use float
    // PNG unsigned char
    typedef float PixelType;
    typedef itk::Image< PixelType, InputDimension >  ImageType;
    typedef itk::ImageFileReader<ImageType> ReaderType;

    typedef std::complex<PixelType> ComplexPixelType;
    typedef itk::Image< ComplexPixelType, InputDimension > ComplexImageType;

    typedef itk::Image<itk::CovariantVector<ImageType::PixelType, InputDimension>,
            InputDimension> VectorImageType;
    typedef itk::Image<itk::CovariantVector<std::complex<ImageType::PixelType>, InputDimension>,
            InputDimension> ComplexVectorImageType;

    typedef float MaskPixelType;
    typedef itk::Image< MaskPixelType, InputDimension > MaskImage;

    typedef unsigned short OutputPixel;
    typedef itk::Image< OutputPixel, InputDimension >  OutputImage;

    static ImageType::Pointer read(std::string image_file_path);
    static MaskImage::Pointer read_mask(std::string image_file_path);

    static ImageType::Pointer perform_masking(const ImageType::Pointer& image,
                                const ImageType::Pointer& mask);

    static ImageType::Pointer bias_field(const ImageType::Pointer& image,
                                         const ImageType::Pointer& mask);

    struct NodePoint{
        int x,y,z;
        NodePoint(int x,int y,int z) : x(x),y(y),z(z) {}

    };

    static ImageType::Pointer create_mask(const ImageType::Pointer& image,
                                          std::vector<NodePoint> node_list);


    static ImageType::Pointer create_mask_by_threshold_and_erosition(
            const ImageType::Pointer& image, float threshold_pixel_value, int erosition_iterations);


    static MaskImage::Pointer create_mask_by_maximum(
            const ImageType::Pointer& image,  const ImageType::Pointer& initial_mask);



    static void write(ImageType::Pointer image, std::string file_path);
    static void write_mask(MaskImage::Pointer image, std::string file_path);

    static ImageType::Pointer histogram(const ImageType::Pointer& image);

    static void histogram_data(const ImageType::Pointer& image,
                                           int bin_count,
                                           ImageType::PixelType window_from,
                                           ImageType::PixelType window_to,
                                           std::vector<double>& intensities,
                                           std::vector<double>& probabilities);

    static void find_min_max_pixel_value(const ImageType::Pointer& image,
                                                float &min_pixel_value,
                                                float &max_pixel_value);

    static void intensity_profile(const ImageType::Pointer & image,
                                  int point1_x, int point1_y,
                                  int point2_x, int point2_y,
                                         std::vector<double>& intensities,
                                         std::vector<double>& distances);

    struct SplineResolution
    {
        int x,y,z;
    };

    static ImageType::Pointer extract_volume(ImageType::Pointer image,
         unsigned int from_x, unsigned int to_x,
         unsigned int from_y, unsigned int to_y,
         unsigned int from_z, unsigned int to_z);

    static ImageType::Pointer backward_difference_x_0_at_boundary(
            ImageType::Pointer f);
    static ImageType::Pointer forward_difference_x_0_at_boundary(
            ImageType::Pointer f);
    static ImageType::Pointer laplace_operator(
            ImageType::Pointer f);
    static ImageType::Pointer laplace_operator_projected(
            ImageType::Pointer f);

    static void gradients_projected_and_laplace(ImageType::Pointer input,
                                                ImageType::Pointer& gradient_x,
                                                ImageType::Pointer& gradient_y,
                                                ImageType::Pointer& laplace);
    static void removeSensorSensitivity(ImageType::Pointer f,
                                        const float alpha,
                                        const float beta,
                                        const int pyramid_levels,
                                        const int iteration_count_factor,
                                        const bool with_max_contraint,
                                        std::function<void(ImageType::Pointer,uint,uint)> iteration_callback,
                                        std::function<void(ImageType::Pointer,ImageType::Pointer)> finished_callback);
    static ImageType::Pointer gammaEnhancement(
                        ImageType::Pointer image,
            ImageType::Pointer illumination, const float gamma);
    static ImageType::Pointer gradient_magnitude_image(ImageType::Pointer input);

    template<typename T>
    static typename T::Pointer clone(const typename T::Pointer image);

    static typename ImageType::Pointer cloneImage(const typename ImageType::Pointer image);

    static ImageType::Pointer bilateralFilter(ImageType::Pointer image,
                                              float sigma_spatial_distance,
                                              float sigma_intensity_distance,
                                              int kernel_size);
    static ImageType::Pointer threshold(ImageType::Pointer image,
                                        ImageType::PixelType lower_threshold_value,
                                        ImageType::PixelType upper_threshold_value,
                                        ImageType::PixelType outside_pixel_value);

    template<typename T2>
    static typename T2::PixelType SumAllPixels(const typename T2::Pointer image);

    static void multiScaleRetinex(ImageType::Pointer image,
            std::vector<MultiScaleRetinex::Scale*> scales,
            std::function<void(ImageType::Pointer)> finished_callback);
private:
    static void removeSensorSensitivity_FFTOperators(const ImageType::SizeType size,
         ImageType::Pointer& fft_laplace,
         ComplexImageType::Pointer& fft_gradient_h,
         ComplexImageType::Pointer& fft_gradient_v);

public:

    struct ReferenceROIStatistic
    {
        float median_value;
        int x;
        int y;
    };

    static ImageType::Pointer splineFit(ImageType::Pointer image, uint spline_order,
                                 uint spline_levels, uint spline_control_points,
                                 std::vector<ReferenceROIStatistic> nodes,
                                        ImageType::Pointer& field_image);
};

#endif // ITKIMAGEPROCESSOR_H
