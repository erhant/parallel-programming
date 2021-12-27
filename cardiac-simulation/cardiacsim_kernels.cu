#include "cardiacsim_kernels.cuh"

// Third Version
__global__ void kernel_v3(double *E, double *E_prev, double *R, const double alpha, const double epsilon,
                          const double dt, const double kk, const double a, const double b, const double M1,
                          const double M2, const int m, const int n) {
  // Setup indexes
  const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int numCols = n + 2;
  const int index = row * numCols + col;

  // Fetch elements
  const double curr = E_prev[index];                                     // current element
  const double left = E_prev[index - 1 + 2 * (col == 1)];                // left element
  const double right = E_prev[index + 1 - 2 * (col == n)];               // right element
  const double top = E_prev[index + numCols * (2 * (row == 1) - 1)];     // top element
  const double bottom = E_prev[index - numCols * (2 * (row == m) - 1)];  // bottom element

  // PDE
  double e = curr + alpha * (right + left - 4 * curr + top + bottom);
  double r = R[index];

  // ODE
  e = e - dt * (kk * e * (e - a) * (e - 1) + e * r);
  E[index] = e;
  R[index] = r + dt * (epsilon + M1 * r / (e + M2)) * (-r - kk * e * (e - b - 1));
}
