#include "kernels.cuh"

// Reduction: Most naive version possible. Everything is in global memory.
__global__ void reduction_kernel(unsigned char* image, int width, int height, int n_pixels, float* sum, float* sum2) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (j >= 0 && j < width && i >= 0 && i < height) {
    long p = (i * width + j);        // pixel index
    float tmp = image[p];            // 1 global access
    atomicAdd(&sum[0], tmp);         // 1 global access
    atomicAdd(&sum2[0], tmp * tmp);  // 1 global access
  }
}
// Statistics: After reduction is finished, sum parameter will be holding the sum basically. The result will be written
// to std_dev.
__global__ void statistics_kernel(float* sum, int n_pixels, float* sum2, float* std_dev) {
  // printf("Sum: %f\tSum2: %f\n", sum[0], sum2[0]);
  float mean = sum[0] / n_pixels;                       // 1 global access
  float variance = (sum2[0] / n_pixels) - mean * mean;  // 1 global access
  std_dev[0] = variance / (mean * mean);                // 1 global access
}

// Compute 1:
__global__ void compute1_kernel(unsigned char* image, int width, int height, int n_pixels, float* std_dev,
                                float* north_deriv, float* south_deriv, float* west_deriv, float* east_deriv,
                                float* diff_coef) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (j > 0 && j < width - 1 && i > 0 && i < height - 1) {
    float gradient_square, num, den, std_dev2, laplacian;
    long k = i * width + j;  // position of current element
    // from global memory to registers
    float std_dev_0 = std_dev[0];  // 1 global access
    float image_k = image[k];      // 1 global access
    float north, south, west, east;
    float diff_coef_k;
    // actual process
    north = image[k - width] -
            image_k;  // north derivative on register --- 1 floating point arithmetic operations, 1 global access
    south = image[k + width] -
            image_k;  // south derivative on register --- 1 floating point arithmetic operations, 1 global access
    west = image[k - 1] -
           image_k;  // west derivative on register --- 1 floating point arithmetic operations, 1 global access
    east = image[k + 1] -
           image_k;          // east derivative on register --- 1 floating point arithmetic operations, 1 global access
    north_deriv[k] = north;  // north derivative on register back to global memory, 1 global access
    south_deriv[k] = south;  // south derivative on register back to global memory, 1 global access
    west_deriv[k] = west;    // west derivative on register back to global memory, 1 global access
    east_deriv[k] = east;    // east derivative on register back to global memory, 1 global access
    gradient_square = (north * north + south * south + west * west + east * east) /
                      (image_k * image_k);                                     // 9 floating point arithmetic operations
    laplacian = (north + south + west + east) / image_k;                       // 4 floating point arithmetic operations
    num = (0.5 * gradient_square) - ((1.0 / 16.0) * (laplacian * laplacian));  // 5 floating point arithmetic operations
    den = 1 + (.25 * laplacian);                                               // 2 floating point arithmetic operations
    std_dev2 = num / (den * den);                                              // 2 floating point arithmetic operations
    den = (std_dev2 - std_dev_0) / (std_dev_0 * (1 + std_dev_0));              // 4 floating point arithmetic operations
    diff_coef_k = 1.0 / (1.0 + den);                                           // 2 floating point arithmetic operations
    if (diff_coef_k < 0) {
      diff_coef_k = 0;
    } else if (diff_coef_k > 1) {
      diff_coef_k = 1;
    }
    diff_coef[k] = diff_coef_k;  // 1 global access
  }
}
// Compute 2:
__global__ void compute2_kernel(float* diff_coef, int width, int height, float lambda, float* north_deriv,
                                float* south_deriv, float* west_deriv, float* east_deriv, unsigned char* image) {
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  __shared__ unsigned char sdata[DIM_THREAD_BLOCK_Y][DIM_THREAD_BLOCK_X + 1];  // +1 to avoid bank conflicts
  float diff_coef_k;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  long k;
  // Load image to shared memory
  if (col < width && row < height) {
    k = row * width + col;  // position of current element
    diff_coef_k = diff_coef[k];
    sdata[ty][tx] = diff_coef_k;
  }
  __syncthreads();
  if (col > 0 && col < width - 1 && row > 0 && row < height - 1) {
    float divergence;
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
      divergence = diff_coef_k * north_deriv[k] + diff_coef[k + width] * south_deriv[k] + diff_coef_k * west_deriv[k] +
                   diff_coef[k + 1] * east_deriv[k];
    } else if (tx == blockDim.x - 1) {
      divergence = diff_coef_k * north_deriv[k] + sdata[ty + 1][tx] * south_deriv[k] + diff_coef_k * west_deriv[k] +
                   diff_coef[k + 1] * east_deriv[k];
    } else if (ty == blockDim.y - 1) {
      divergence = diff_coef_k * north_deriv[k] + diff_coef[k + width] * south_deriv[k] + diff_coef_k * west_deriv[k] +
                   sdata[ty][tx + 1] * east_deriv[k];
    } else {
      divergence = diff_coef_k * north_deriv[k] + sdata[ty + 1][tx] * south_deriv[k] + diff_coef_k * west_deriv[k] +
                   sdata[ty][tx + 1] * east_deriv[k];
    }
    image[k] =
        image[k] +
        0.25 * lambda * divergence;  // --- 3 floating point arithmetic operations, 2 global access (read and write)
  }
}

