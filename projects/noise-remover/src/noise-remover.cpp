#include "noise-remover.hpp"

void driver_function() {
  // (HOST) Part I: Allocate and initialize variables
  double time_0, time_1, time_2, time_3, time_4, time_5, time_6, time_7, time_8;  // time variables
  time_0 = get_time();
  const char* filename = "input.pgm";
  const char* outputname = "output.png";
  int width, height, pixelWidth, n_pixels;
  int n_iter = 50;
  float lambda = 0.5;
  float *d_north_deriv, *d_south_deriv, *d_west_deriv, *d_east_deriv;  // directional derivatives (TO DEVICE)
  float* d_diff_coef;                                                  // diffusion coefficient (TO DEVICE)
  time_1 = get_time();
  cudaStream_t stream;

  // (HOST) Part II: Parse command line arguments
  if (argc < 2) {
    printf("Usage: %s [-i < filename>] [-iter <n_iter>] [-l <lambda>] [-o <outputfilename>]\n", argv[0]);
    return (-1);
  }
  for (int ac = 1; ac < argc; ac++) {
    if (MATCH_ARG("-i")) {
      filename = argv[++ac];
    } else if (MATCH_ARG("-iter")) {
      n_iter = atoi(argv[++ac]);
    } else if (MATCH_ARG("-l")) {
      lambda = atof(argv[++ac]);
    } else if (MATCH_ARG("-o")) {
      outputname = argv[++ac];
    } else {
      printf("Usage: %s [-i < filename>] [-iter <n_iter>] [-l <lambda>] [-o <outputfilename>]\n", argv[0]);
      return (-1);
    }
  }
  time_2 = get_time();

  // (HOST) Part III: Read image
  printf("Reading image...\n");
  unsigned char* image = stbi_load(filename, &width, &height, &pixelWidth, 0);
  if (!image) {
    fprintf(stderr, "Couldn't load image.\n");
    return (-1);
  }
  printf("Image Read. Width : %d, Height : %d, nComp: %d\n", width, height, pixelWidth);
  n_pixels = height * width;  // W * H is the number of pixels
  time_3 = get_time();

  // (HOST & DEVICE) Part IV: Allocate variables
  unsigned char* d_image;
  float* d_sum;
  float* d_sum2;
  float* d_std_dev;
  CUDA_CHECK(cudaMalloc(&d_image, n_pixels * sizeof(unsigned char)));  // image is also allocated there
  CUDA_CHECK(cudaMalloc(&d_north_deriv, n_pixels * sizeof(float)));    // north derivative allocate on device
  CUDA_CHECK(cudaMalloc(&d_south_deriv, n_pixels * sizeof(float)));    // south derivative allocate on device
  CUDA_CHECK(cudaMalloc(&d_west_deriv, n_pixels * sizeof(float)));     // west derivative allocate on device
  CUDA_CHECK(cudaMalloc(&d_east_deriv, n_pixels * sizeof(float)));     // east derivative allocate on device
  CUDA_CHECK(
      cudaMalloc(&d_diff_coef, n_pixels * sizeof(float)));  // diffusion coefficient derivative allocate on device
  CUDA_CHECK(cudaMalloc(&d_sum, 1 * sizeof(float)));        // scalar
  CUDA_CHECK(cudaMalloc(&d_sum2, 1 * sizeof(float)));       // scalar
  CUDA_CHECK(cudaMalloc(&d_std_dev, 1 * sizeof(float)));    // scalar

  // (HOST) Extra: Calculate block and grid parameters.
  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1);
  dim3 grid((unsigned int)ceil(((float)width) / ((float)block.x)),
            (unsigned int)ceil(((float)height) / ((float)block.y)), 1);  // +1 blocks for fringing

  printf("\n\tBlocks per grid: (%d, %d, 1)\n\tThreads per block: (%d, %d, 1)\n\n", grid.x, grid.y, block.x, block.y);

  // (DEVICE) Extra: Copy everything to device.
  CUDA_CHECK(cudaStreamCreate(&stream));                                                             // create stream
  CUDA_CHECK(cudaMemcpy(d_image, image, n_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice));  // transfer image
  time_4 = get_time();

  // (DEVICE) Part V: Compute --- n_iter * (3 * height * width + 42 * (height-1) * (width-1) + 6) floating point
  // arithmetic operations in totaL
  for (int iter = 0; iter < n_iter; iter++) {
    CUDA_CHECK(cudaMemsetAsync(d_sum, 0, 1 * sizeof(float), stream));   // set sum 0
    CUDA_CHECK(cudaMemsetAsync(d_sum2, 0, 1 * sizeof(float), stream));  // set sum2 0
    reduction_kernel<<<grid, block, 0, stream>>>(d_image, width, height, n_pixels, d_sum, d_sum2);
    CUDA_GET_LAST_ERR(1);
    statistics_kernel<<<1, 1, 0, stream>>>(d_sum, n_pixels, d_sum2, d_std_dev);
    CUDA_GET_LAST_ERR(2);
    compute1_kernel<<<grid, block, 0, stream>>>(d_image, width, height, n_pixels, d_std_dev, d_north_deriv,
                                                d_south_deriv, d_west_deriv, d_east_deriv, d_diff_coef);
    CUDA_GET_LAST_ERR(3);
    compute2_kernel<<<grid, block, 0, stream>>>(d_diff_coef, width, height, lambda, d_north_deriv, d_south_deriv,
                                                d_west_deriv, d_east_deriv, d_image);
    CUDA_GET_LAST_ERR(4);
  }
  // (DEVICE) Extra: Retrieve image from device
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaStreamDestroy(stream));                                                             // destroy streams
  CUDA_CHECK(cudaMemcpy(image, d_image, n_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost));  // transfer image
  time_5 = get_time();

  // (DEVICE & HOST) Part VI: Write image to file
  stbi_write_png(outputname, width, height, pixelWidth, image, 0);
  time_6 = get_time();

  // (HOST) Part VII: Get the average of sum of pixels for testing and calculate GFLOPS
  // FOR VALIDATION - DO NOT PARALLELIZE
  float test = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      test += image[i * width + j];
    }
  }
  test /= n_pixels;

  float gflops =
      (float)(n_iter * 1E-9 * (3 * height * width + 42 * (height - 1) * (width - 1) + 6)) / (time_5 - time_4);
  time_7 = get_time();

  // (HOST & DEVICE) Part VII: Deallocate variables
  stbi_image_free(image);
  CUDA_CHECK(cudaFree(d_image));        // device's image
  CUDA_CHECK(cudaFree(d_north_deriv));  // north derivative allocate on device
  CUDA_CHECK(cudaFree(d_south_deriv));  // south derivative allocate on device
  CUDA_CHECK(cudaFree(d_west_deriv));   // west derivative allocate on device
  CUDA_CHECK(cudaFree(d_east_deriv));   // east derivative allocate on device
  CUDA_CHECK(cudaFree(d_diff_coef));    // diffusion coefficient derivative allocate on device
  CUDA_CHECK(cudaFree(d_sum));          // scalar
  CUDA_CHECK(cudaFree(d_sum2));         // scalar
  CUDA_CHECK(cudaFree(d_std_dev));      // scalar
  time_8 = get_time();

  // (HOST) Final: Print
  printf("Time spent in different stages of the application:\n");
  printf("%9.6f s => Part 1: allocate and initialize variables\n", (time_1 - time_0));
  printf("%9.6f s => Part 2: parse command line arguments\n", (time_2 - time_1));
  printf("%9.6f s => Part 3: read image\n", (time_3 - time_2));
  printf("%9.6f s => Part 4: allocate variables\n", (time_4 - time_3));
  printf("%9.6f s => Part 5: compute\n", (time_5 - time_4));
  printf("%9.6f s => Part 6: write image to file\n", (time_6 - time_5));
  printf("%9.6f s => Part 7: get average of sum of pixels for testing and calculate GFLOPS\n", (time_7 - time_6));
  printf("%9.6f s => Part 8: deallocate variables\n", (time_7 - time_6));
  printf("Total time: %9.6f s\n", (time_8 - time_0));
  printf("Average of sum of pixels: %9.6f\n", test);
  printf("GFLOPS: %f\n", gflops);
}