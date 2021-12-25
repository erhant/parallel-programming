#include "cardiacsim_kernels.cuh"

// Third Version
__global__ void kernel_v3(double *E, double *E_prev, double *R,
                          const double alpha, const double epsilon,
                          const double dt, const double kk, const double a,
                          const double b, const double M1, const double M2,
                          const int m, const int n) {
  const int numCols = n + 2;
  const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int index = row * numCols + col;

  const double curr = E_prev[index];                       // current index
  const double left = E_prev[index - 1 + 2 * (col == 1)];  // left index
  const double right = E_prev[index + 1 - 2 * (col == n)]; // right index
  const double up = E_prev[index + numCols * (2 * (row == 1) - 1)]; // up index
  const double down =
      E_prev[index - numCols * (2 * (row == m) - 1)]; // down index

  // PDE
  const double e = curr + alpha * (right + left - 4 * curr + up + down);
  const double r = R[index];

  // ODE
  E[index] = e - dt * (kk * e * (e - a) * (e - 1) + e * r);
  R[index] =
      r + dt * (epsilon + M1 * r / (e + M2)) * (-r - kk * e * (e - b - 1));
}

extern "C" double *cudalloc2D(int m, int n) {
  double *E;
  cudaMalloc((void **)&E, sizeof(double) * m * n);
  assert(E);
  return (E);
}

extern "C" void freecuda(double **E) { cudaFree(E); }

extern "C" void copyToDevice(int m, int n, double *d_E, double *d_E_prev,
                             double *d_R, double **E, double **E_prev,
                             double **R) {

  int size = sizeof(double) * (n + 2) * (m + 2);
  cudaMemcpy(d_E, &E[m + 2], size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_E_prev, &E_prev[m + 2], size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_R, &R[m + 2], size, cudaMemcpyHostToDevice);
}

extern "C" void copyFromDevice(int m, int n, double *d_E, double *d_E_prev,
                               double *d_R, double **E, double **E_prev,
                               double **R) {

  int size = sizeof(double) * (n + 2) * (m + 2);
  cudaMemcpy(&E[m + 2], d_E, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(&E_prev[m + 2], d_E_prev, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(&R[m + 2], d_R, size, cudaMemcpyDeviceToHost);
}

double simulate_GPU(double **E, double **E_prev, double **R,
                    const double epsilon, const double dt, const double kk,
                    const double a, const double b, const double M1,
                    const double M2, const double alpha, const int m,
                    const int n, double *d_E, double *d_E_prev, double *d_R,
                    int kernel, int bx, int by) {

  return 0;
}
