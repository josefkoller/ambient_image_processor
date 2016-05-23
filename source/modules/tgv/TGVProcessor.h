#ifndef TGVPROCESSOR_H
#define TGVPROCESSOR_H

#include <ITKImage.h>
#include "ThrustImage.cuh"


class TGVProcessor
{
private:
    TGVProcessor();

public:

    typedef std::function<void(uint iteration_index, uint iteration_count,
                               HostThrustImage* u)> HostIterationFinished;
    typedef std::function<void(uint iteration_index, uint iteration_count,
                               DeviceThrustImage* u)> DeviceIterationFinished;

    typedef std::function<void(uint iteration_index, uint iteration_count,
                               ITKImage u)> IterationFinished;

    template<typename Pixel>
    using IterationCallback = std::function<void(uint iteration_index, uint iteration_count, Pixel* u)>;

private:
    template<typename ThrustImage>
    static ThrustImage* convert(ITKImage itk_image);

    template<typename ThrustImage>
    static ITKImage convert(ThrustImage* image);

    template<typename Pixel>
    using TGVAlgorithm = std::function<Pixel*(Pixel* f, IterationCallback<Pixel> iteration_callback)>;
public:

    static ITKImage processTVL2GPUThrust(ITKImage input_image,
      const Pixel lambda, const uint iteration_count,
      const uint paint_iteration_interval, IterationFinished iteration_finished_callback);
    static ITKImage processTVL2CPU(ITKImage input_image,
      const Pixel lambda, const uint iteration_count,
      const uint paint_iteration_interval,
      IterationFinished iteration_finished_callback);

    static ITKImage processTVGPUCuda(ITKImage input_image,
                                IterationFinished iteration_finished_callback,
                                TGVAlgorithm<Pixel> tgv_algorithm);

    static ITKImage processTVL2GPUCuda(ITKImage input_image,
      const Pixel lambda,
      const Pixel alpha0,
      const Pixel alpha1,
      const uint iteration_count,
      const uint paint_iteration_interval, IterationFinished iteration_finished_callback);

    static ITKImage processTVL1GPUCuda(ITKImage input_image,
      const Pixel lambda,
      const Pixel alpha0,
      const Pixel alpha1,
      const uint iteration_count,
      const uint paint_iteration_interval, IterationFinished iteration_finished_callback);

    static ITKImage processTGV2L1GPUCuda(ITKImage input_image,
      const Pixel lambda,
      const Pixel alpha0,
      const Pixel alpha1,
      const uint iteration_count,
      const uint paint_iteration_interval, IterationFinished iteration_finished_callback);

    static ITKImage processTGV2L2GPUCuda(ITKImage input_image,
      const Pixel lambda,
      const Pixel alpha0,
      const Pixel alpha1,
      const uint iteration_count,
      const uint paint_iteration_interval, IterationFinished iteration_finished_callback);
};

#endif // TGVPROCESSOR_H
