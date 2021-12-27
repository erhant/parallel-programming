/*
 * Solves the Panfilov model using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * and reimplementation by Scott B. Baden, UCSD
 *
 * Modified and  restructured by Didem Unat, Koc University
 *
 * Refer to "Detailed Numerical Analyses of the Aliev-Panfilov Model on GPGPU"
 * https://www.simula.no/publications/detailed-numerical-analyses-aliev-panfilov-model-gpgpu
 * by Xing Cai, Didem Unat and Scott Baden
 *
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <iomanip>
#include <iostream>

#include "cardiacsim_kernels.cuh"

using namespace std;

static const double kMicro = 1.0e-6;

void cmdLine(int argc, char *argv[], double &T, int &n, int &px, int &py, int &plot_freq, int &kernel_no);

extern double getTime() {
  struct timeval TV;
  struct timezone TZ;

  const int RC = gettimeofday(&TV, &TZ);
  if (RC == -1) {
    cerr << "ERROR: Bad call to gettimeofday" << endl;
    exit(-1);
  }

  return (((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec));
}

double **alloc2D(int m, int n) {
  double **E;
  int nx = n, ny = m;
  E = (double **)malloc(sizeof(double *) * ny + sizeof(double) * nx * ny);
  assert(E);
  int j;
  for (j = 0; j < ny; j++) E[j] = (double *)(E + ny) + j * nx;
  return (E);
}

double stats(double **E, int m, int n, double *_mx) {
  double mx = -1;
  double l2norm = 0;
  int i, j;
  for (j = 1; j <= m; j++)
    for (i = 1; i <= n; i++) {
      l2norm += E[j][i] * E[j][i];
      if (E[j][i] > mx) mx = E[j][i];
    }
  *_mx = mx;
  l2norm /= (double)((m) * (n));
  l2norm = sqrt(l2norm);
  return l2norm;
}

double simulate_CPU(double **E, double **E_prev, double **R, const double alpha, const int n, const int m,
                    const double kk, const double dt, const double a, const double epsilon, const double M1,
                    const double M2, const double b, double *d_E, double *d_E_prev, double *d_R) {
  int i, j;

  /*
   * Copy data from boundary of the computational box to the padding region,
   * set up for differencing on the boundary of the computational box Using mirror boundaries
   */
  for (j = 1; j <= m; j++) E_prev[j][0] = E_prev[j][2];
  for (j = 1; j <= m; j++) E_prev[j][n + 1] = E_prev[j][n - 1];

  for (i = 1; i <= n; i++) E_prev[0][i] = E_prev[2][i];
  for (i = 1; i <= n; i++) E_prev[m + 1][i] = E_prev[m - 1][i];

  // Solve for the excitation, the PDE
  for (j = 1; j <= m; j++)
    for (i = 1; i <= n; i++)
      E[j][i] = E_prev[j][i] +
                alpha * (E_prev[j][i + 1] + E_prev[j][i - 1] - 4 * E_prev[j][i] + E_prev[j + 1][i] + E_prev[j - 1][i]);

  // Solve the ODE, advancing excitation and recovery to the next timtestep
  for (j = 1; j <= m; j++)
    for (i = 1; i <= n; i++)
      E[j][i] = E[j][i] - dt * (kk * E[j][i] * (E[j][i] - a) * (E[j][i] - 1) + E[j][i] * R[j][i]);

  for (j = 1; j <= m; j++)
    for (i = 1; i <= n; i++)
      R[j][i] =
          R[j][i] + dt * (epsilon + M1 * R[j][i] / (E[j][i] + M2)) * (-R[j][i] - kk * E[j][i] * (E[j][i] - b - 1));

  return 0;
}

int main(int argc, char **argv) {
  double **E;       // E is the "Excitation" variable, a voltage
  double **R;       // R is the "Recovery" variable
  double **E_prev;  // E_prev is the Excitation variable for the previous timestep, and is used in time integration

  // Various constants - these definitions shouldn't change
  const double a = 0.1, b = 0.1, kk = 8.0, M1 = 0.07, M2 = 0.3, epsilon = 0.01, d = 5e-5;

  double T = 1000.0;
  int m = 200, n = 200;
  int plot_freq = 0;
  int bx = 1, by = 1;
  int kernel = 1;

  cmdLine(argc, argv, T, n, bx, by, plot_freq, kernel);
  m = n;
  // Allocate contiguous memory for solution arrays
  // The computational box is defined on [1:m+1,1:n+1]
  // We pad the arrays in order to facilitate differencing on the
  // boundaries of the computation box
  E = alloc2D(m + 2, n + 2);
  E_prev = alloc2D(m + 2, n + 2);
  R = alloc2D(m + 2, n + 2);

  // Initialization
  int i, j;
  for (j = 1; j <= m; j++)
    for (i = 1; i <= n; i++) E_prev[j][i] = R[j][i] = 0;

  for (j = 1; j <= m; j++)
    for (i = n / 2 + 1; i <= n; i++) E_prev[j][i] = 1.0;

  for (j = m / 2 + 1; j <= m; j++)
    for (i = 1; i <= n; i++) R[j][i] = 1.0;

  // Copy stuff to device
  const int memcpy_size = sizeof(double) * (n + 2) * (m + 2);
  double *d_E, *d_R, *d_E_prev;
  CUDA_CHECK_CALL(cudaMalloc(&d_E, memcpy_size));
  CUDA_CHECK_CALL(cudaMalloc(&d_E_prev, memcpy_size));
  CUDA_CHECK_CALL(cudaMalloc(&d_R, memcpy_size));
  CUDA_CHECK_CALL(cudaMemcpy(d_E, &E[m + 2], memcpy_size, cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_E_prev, &E_prev[m + 2], memcpy_size, cudaMemcpyHostToDevice));
  CUDA_CHECK_CALL(cudaMemcpy(d_R, &R[m + 2], memcpy_size, cudaMemcpyHostToDevice));

  // For time integration, these values shouldn't change
  double dx = 1.0 / n;
  double rp = kk * (b + 1) * (b + 1) / 4;
  double dte = (dx * dx) / (d * 4 + ((dx * dx)) * (rp + kk));
  double dtr = 1 / (epsilon + ((M1 / M2) * rp));
  double dt = (dte < dtr) ? 0.95 * dte : 0.95 * dtr;
  double alpha = d * dt / (dx * dx);

  // Print configuration
  cout << "Grid Size       : " << n << endl;
  cout << "Duration        : " << T << endl;
  cout << "Time step dt    : " << dt << endl;
  cout << "Block Size      : " << bx << " x " << by << endl;
  cout << "Using CUDA Kernel Version: " << 3 << endl;
  cout << endl;

  // Prepare block sizes
  const int xBlocks = m / bx;
  const int yBlocks = n / by;
  const dim3 tpb(bx, by);            // threads per block
  const dim3 bpg(xBlocks, yBlocks);  // blocks per grid

  // Start the timer
  const double t0 = getTime();
  double t = 0.0;
  int niter = 0;
  bool isEven = true;  // we start with 0th step, which is even
  while (t < T) {
    t += dt;
    niter++;

    // run kernel
    kernel_v3<<<bpg, tpb>>>(d_E, d_E_prev, d_R, alpha, epsilon, dt, kk, a, b, M1, M2, m, n);
    isEven = !isEven;

    // instead of synchronizing and swapping, give swapped params to the kernel
    if (t < T) {
      t += dt;
      niter++;

      // this if check is negligible due to the overlap of the kernel computation
      kernel_v3<<<bpg, tpb>>>(d_E_prev, d_E, d_R, alpha, epsilon, dt, kk, a, b, M1, M2, m, n);
      // CUDA_CHECK_LAST("KERNEL 3");
      // CUDA_CHECK_CALL(cudaDeviceSynchronize());
      isEven = !isEven;
    }
  }

  CUDA_CHECK_CALL(cudaDeviceSynchronize());
  double time_elapsed = getTime() - t0;

  // Copy stuff back (with respect to isEven)
  CUDA_CHECK_CALL(cudaMemcpy(&E[m + 2], isEven ? d_E : d_E_prev, memcpy_size, cudaMemcpyDeviceToHost));
  CUDA_CHECK_CALL(cudaMemcpy(&E_prev[m + 2], isEven ? d_E_prev : d_E, memcpy_size, cudaMemcpyDeviceToHost));
  CUDA_CHECK_CALL(cudaMemcpy(&R[m + 2], d_R, memcpy_size, cudaMemcpyDeviceToHost));

  double Gflops = (double)(niter * (1E-9 * n * n) * 28.0) / time_elapsed;
  double BW = (double)(niter * 1E-9 * (n * n * sizeof(double) * 4.0)) / time_elapsed;

  cout << "Number of Iterations        : " << niter << endl;
  cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
  cout << "Sustained Gflops Rate       : " << Gflops << endl;
  cout << "Sustained Bandwidth (GB/sec): " << BW << endl << endl;

  double maxval;
  double l2norm = stats(E_prev, m, n, &maxval);
  cout << "Max: " << maxval << "\tL2norm: " << l2norm << endl;

  // Frees
  free(E);
  free(E_prev);
  free(R);
  CUDA_CHECK_CALL(cudaFree(d_E));
  CUDA_CHECK_CALL(cudaFree(d_E_prev));
  CUDA_CHECK_CALL(cudaFree(d_R));

  return 0;
}

void cmdLine(int argc, char *argv[], double &T, int &n, int &bx, int &by, int &plot_freq, int &kernel) {
  /// Command line arguments
  // Default value of the domain sizes
  static struct option long_options[] = {
      {"n", required_argument, 0, 'n'},
      {"bx", required_argument, 0, 'x'},
      {"by", required_argument, 0, 'y'},
      {"tfinal", required_argument, 0, 't'},
  };
  // Process command line arguments
  int ac;
  for (ac = 1; ac < argc; ac++) {
    int c;
    while ((c = getopt_long(argc, argv, "n:x:y:t:", long_options, NULL)) != -1) {
      switch (c) {
        // Size of the computational box
        case 'n':
          n = atoi(optarg);
          break;
        // X block geometry
        case 'x':
          bx = atoi(optarg);
          break;
        // Y block geometry
        case 'y':
          by = atoi(optarg);
          break;
        // Length of simulation, in simulated time units
        case 't':
          T = atof(optarg);
          break;
        // Error
        default:
          printf(
              "Usage: \n"
              "\t-n <domain size>\n"
              "\t-t <final time >\n"
              "\t-x <blockDim.x>\n"
              "\t-y <blockDim.y>\n");
          exit(-1);
      }
    }
  }
}
