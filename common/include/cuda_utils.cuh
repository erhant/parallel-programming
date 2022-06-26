#pragma once

#include <assert.h>
#ifdef __NVCC__
#include <cuda.h>
#endif

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include <cfloat>

// Taken from https://github.com/NVIDIA/multi-gpu-programming-models/blob/master/single_gpu/jacobi.cu
#define CUDA_CHECK_CALL(call)                                                                                     \
  {                                                                                                               \
    cudaError_t cudaStatus = call;                                                                                \
    if (cudaSuccess != cudaStatus)                                                                                \
      fprintf(stderr, "error CUDA RT call \"%s\" in line %d of file %s failed with  %s (%d).\n", #call, __LINE__, \
              __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);                                              \
  }

// Check the success of a kernel. Since they dont return an error code, we check with cudaGetLastError instead.
#define CUDA_GET_LAST_ERR(label)                                          \
  {                                                                       \
    cudaError_t err = cudaGetLastError();                                 \
    if (cudaSuccess != err) {                                             \
      fprintf(stderr, "%s failed: %s\n", label, cudaGetErrorString(err)); \
    }                                                                     \
  }

// Start CUDA events for the stopwatch
#define START_GPU_TIMERS()                              \
  cudaEvent_t start_cudaEvent, stop_cudaEvent;          \
  CUDA_CHECK_CALL(cudaEventCreate(&(start_cudaEvent))); \
  CUDA_CHECK_CALL(cudaEventCreate(&(stop_cudaEvent)));  \
  CUDA_CHECK_CALL(cudaEventRecord(start_cudaEvent, 0));

// Stop CUDA events for the stopwatch, and record the milliseconds passed
#define STOP_GPU_TIMERS(ms)                                                      \
  CUDA_CHECK_CALL(cudaEventRecord(stop_cudaEvent, 0));                           \
  CUDA_CHECK_CALL(cudaEventSynchronize(stop_cudaEvent));                         \
  CUDA_CHECK_CALL(cudaEventElapsedTime(&(ms), start_cudaEvent, stop_cudaEvent)); \
  CUDA_CHECK_CALL(cudaEventDestroy(start_cudaEvent));                            \
  CUDA_CHECK_CALL(cudaEventDestroy(stop_cudaEvent));