#include "cardiacsim_kernels.cuh"

// Third Version
__global__ void kernel_v3(double *E, double *E_prev, double *R, const double alpha, const double epsilon,
                          const double dt, const double kk, const double a, const double b, const double M1,
                          const double M2, const int m, const int n) {
  // indexing setup
  const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int numCols = n + 2;
  const int index = row * numCols + col;

  // get elements
  const double curr = E_prev[index];                                   // current index
  const double left = E_prev[index - 1 + 2 * (col == 1)];              // left index
  const double right = E_prev[index + 1 - 2 * (col == n)];             // right index
  const double up = E_prev[index + numCols * (2 * (row == 1) - 1)];    // up index
  const double down = E_prev[index - numCols * (2 * (row == m) - 1)];  // down index

  // PDE
  const double e = curr + alpha * (right + left + up + down - 4 * curr);
  const double r = R[index];

  // ODE
  E[index] = e - dt * (kk * e * (e - a) * (e - 1) + e * r);
  R[index] = r + dt * (epsilon + M1 * r / (e + M2)) * (-r - kk * e * (e - b - 1));
}
