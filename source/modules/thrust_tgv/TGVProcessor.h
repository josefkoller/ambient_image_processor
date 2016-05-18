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
private:
    template<typename ThrustImage>
    static ThrustImage* convert(ITKImage itk_image);

    template<typename ThrustImage>
    static ITKImage convert(ThrustImage* image);
public:

    static ITKImage processTVL2GPU(ITKImage input_image,
      const Pixel lambda, const uint iteration_count, IterationFinished iteration_finished_callback);
    static ITKImage processTVL2CPU(ITKImage input_image,
      const Pixel lambda, const uint iteration_count, IterationFinished iteration_finished_callback);
};

#endif // TGVPROCESSOR_H
