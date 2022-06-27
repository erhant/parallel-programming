#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 32

__global__ void reduction_kernel(unsigned char* image, int width, int height, int n_pixels, float* sum, float* sum2);

__global__ void statistics_kernel(float* sum, int n_pixels, float* sum2, float* std_dev);

__global__ void compute1_kernel(unsigned char* image, int width, int height, int n_pixels, float* std_dev,
  float* north_deriv, float* south_deriv, float* west_deriv, float* east_deriv,
  float* diff_coef);

  __global__ void compute2_kernel(float* diff_coef, int width, int height, float lambda, float* north_deriv,
    float* south_deriv, float* west_deriv, float* east_deriv, unsigned char* image);