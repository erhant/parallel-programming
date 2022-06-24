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
// double start = omp_get_wtime(), stop;
#define START_TIMERS()                        \
  cudaEvent_t start, stop;                    \
  CUDA_CHECK_CALL(cudaEventCreate(&(start))); \
  CUDA_CHECK_CALL(cudaEventCreate(&(stop)));  \
  CUDA_CHECK_CALL(cudaEventRecord(start, 0));

// Stop CUDA events for the stopwatch, and record the milliseconds passed.
// CUDA_CHECK_CALL(cudaDeviceSynchronize());
// stop = omp_get_wtime();
// ms = (stop - start) / 1000.0;
#define STOP_TIMERS(ms)                                          \
  CUDA_CHECK_CALL(cudaEventRecord(stop, 0));                     \
  CUDA_CHECK_CALL(cudaEventSynchronize(stop));                   \
  CUDA_CHECK_CALL(cudaEventElapsedTime(&(ms), (start), (stop))); \
  CUDA_CHECK_CALL(cudaEventDestroy(start));                      \
  CUDA_CHECK_CALL(cudaEventDestroy(stop));