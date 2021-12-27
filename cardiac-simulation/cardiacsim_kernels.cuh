#ifndef __CARDIACSIM_KERNELS_H_
#define __CARDIACSIM_KERNELS_H_

// Check a CUDA function call that returns an error
#define CUDA_CHECK_CALL(call)                                                                                     \
  {                                                                                                               \
    cudaError_t cudaStatus = call;                                                                                \
    if (cudaSuccess != cudaStatus)                                                                                \
      fprintf(stderr, "error CUDA RT call \"%s\" in line %d of file %s failed with  %s (%d).\n", #call, __LINE__, \
              __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);                                              \
  }

#define CUDA_CHECK_LAST(label)                                            \
  {                                                                       \
    cudaError_t err = cudaGetLastError();                                 \
    if (cudaSuccess != err) {                                             \
      fprintf(stderr, "%s failed: %s\n", label, cudaGetErrorString(err)); \
    }                                                                     \
  }

__global__ void kernel_v3(double *E, double *E_prev, double *R, const double alpha, const double epsilon,
                          const double dt, const double kk, const double a, const double b, const double M1,
                          const double M2, const int m, const int n);

#endif  // __CARDIACSIM_KERNELS_H_