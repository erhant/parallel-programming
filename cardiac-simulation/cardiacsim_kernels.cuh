#ifndef __CARDIACSIM_KERNELS_H_
#define __CARDIACSIM_KERNELS_H_

#include <cassert>
#include <stdio.h>

__global__ void kernel_v3(double *E, double *E_prev, double *R,
                          const double alpha, const double epsilon,
                          const double dt, const double kk, const double a,
                          const double b, const double M1, const double M2,
                          int m, int n);
z

#endif // __CARDIACSIM_KERNELS_H_