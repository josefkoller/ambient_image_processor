#ifndef CUDA_HELPER
#define CUDA_HELPER

#include <cuda.h>
#include <stdio.h>

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t error_code, const char *file, int line, bool abort=true)
{
   if (error_code != cudaSuccess)
   {
      printf("cuda error: %s %s %d\n", cudaGetErrorString(error_code), file, line);
      if (abort)
      {
          exit(error_code);
      }
   }
}

T* copyToDevice<T>(T* host_data, size_t size)
{
    T* device_data;
    cudaCheckError( cudaMalloc(&device_data, size) );
    cudaCheckError( cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice) );
}

T* copyToHost<T>(T* device_data, unsigned int array_length)
    {
    T* host_data = new T[array_length];
    cudaCheckError( cudaMemcpy(host_data, device_data, sizeof(T) * array_length, cudaMemcpyDeviceToHost) );
}

#endif //CUDA_HELPER
