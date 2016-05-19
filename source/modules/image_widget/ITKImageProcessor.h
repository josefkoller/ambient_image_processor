#ifndef ITKIMAGEPROCESSOR_H
#define ITKIMAGEPROCESSOR_H

#include <itkImage.h>
#include <itkImageFileReader.h>

#include <functional>
#include <vector>
#include <complex>

const unsigned int InputDimension = 2;

class ITKImageProcessor
{
public:
    // MHA use ImageType::PixelType
    // PNG unsigned char
    typedef double PixelType;
    typedef itk::Image< PixelType, InputDimension >  ImageType;
    typedef itk::ImageFileReader<ImageType> ReaderType;

    typedef std::complex<PixelType> ComplexPixelType;
    typedef itk::Image< ComplexPixelType, InputDimension > ComplexImageType;

    typedef itk::Image<itk::CovariantVector<ImageType::PixelType, InputDimension>,
            InputDimension> VectorImageType;
    typedef itk::Image<itk::CovariantVector<std::complex<ImageType::PixelType>, InputDimension>,
            InputDimension> ComplexVectorImageType;

    typedef ImageType::PixelType MaskPixelType;
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
            const ImageType::Pointer& image, ImageType::PixelType threshold_pixel_value, int erosition_iterations);


    static MaskImage::Pointer create_mask_by_maximum(
            const ImageType::Pointer& image,  const ImageType::Pointer& initial_mask);



    static void write(ImageType::Pointer image, std::string file_path);
    static void write_mask(MaskImage::Pointer image, std::string file_path);

    static ImageType::Pointer histogram(const ImageType::Pointer& image);


    static void find_min_max_pixel_value(const ImageType::Pointer& image,
                                                ImageType::PixelType &min_pixel_value,
                                                ImageType::PixelType &max_pixel_value);


    struct SplineResolution
    {
        int x,y,z;
    };


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
                                        const ImageType::PixelType alpha,
                                        const ImageType::PixelType beta,
                                        const int pyramid_levels,
                                        const int iteration_count_factor,
                                        const bool with_max_contraint,
                                        std::function<void(ImageType::Pointer,uint,uint)> iteration_callback,
                                        std::function<void(ImageType::Pointer,ImageType::Pointer)> finished_callback);
    static ImageType::Pointer gammaEnhancement(
                        ImageType::Pointer image,
            ImageType::Pointer illumination, const ImageType::PixelType gamma);
    static ImageType::Pointer gradient_magnitude_image(ImageType::Pointer input);

    template<typename T>
    static typename T::Pointer clone(const typename T::Pointer image);

    static typename ImageType::Pointer cloneImage(const typename ImageType::Pointer image);

    static ImageType::Pointer bilateralFilter(ImageType::Pointer image,
                                              ImageType::PixelType sigma_spatial_distance,
                                              ImageType::PixelType sigma_intensity_distance,
                                              int kernel_size);
    static ImageType::Pointer threshold(ImageType::Pointer image,
                                        ImageType::PixelType lower_threshold_value,
                                        ImageType::PixelType upper_threshold_value,
                                        ImageType::PixelType outside_pixel_value);

    template<typename T2>
    static typename T2::PixelType SumAllPixels(const typename T2::Pointer image);

private:
    static void removeSensorSensitivity_FFTOperators(const ImageType::SizeType size,
         ImageType::Pointer& fft_laplace,
         ComplexImageType::Pointer& fft_gradient_h,
         ComplexImageType::Pointer& fft_gradient_v);

};

#endif // ITKIMAGEPROCESSOR_H
