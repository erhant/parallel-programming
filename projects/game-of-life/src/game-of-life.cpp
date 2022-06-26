#include "game-of-life.hpp"

static GOF_Plotter *plotter = NULL;

void game_of_life_serial(const int nx, const int ny, const int numthreads, const int maxiter, int population,
                         char **currWorld, char **nextWorld) {
  char **tmpWorld = NULL;
  int i, j, iter;
  int sum = 0;
  short proceedToSwap;  // flag variable
  double t0;

  for (iter = 0; iter < maxiter && population != 0; iter++) {
    population = 0;
    for (int i = 1; i < nx - 1; i++) {
      for (int j = 1; j < ny - 1; j++) {
        // calculate neighbor count
        int nn = currWorld[i + 1][j] + currWorld[i - 1][j] + currWorld[i][j + 1] + currWorld[i][j - 1] +
                 currWorld[i + 1][j + 1] + currWorld[i - 1][j - 1] + currWorld[i - 1][j + 1] + currWorld[i + 1][j - 1];
        // if alive check if you die, if dead check if you can produce.
        nextWorld[i][j] = currWorld[i][j] ? (nn == 2 || nn == 3) : (nn == 3);
        // update population
        population += nextWorld[i][j];
      }
    }

    // swap pointers
    tmpWorld = nextWorld;
    nextWorld = currWorld;
    currWorld = tmpWorld;

    if (plotter) plotter->plot(iter, population, currWorld);
  }
}

void game_of_life_parallel(const int nx, const int ny, const int numthreads, const int maxiter, int population,
                           char **currWorld, char **nextWorld) {
  char **tmpWorld = NULL;
  int i, j, iter;
  int sum = 0;
  short proceedToSwap;  // flag variable
  double t0;

  // do one iteration outside
#pragma omp parallel num_threads(numthreads)
  {
#pragma omp for reduction(+: sum) private(j) // we could use collapse(2) but then it would be 2D decomposition, so we go for only one loop parallelization.
    for (i = 1; i < nx - 1; i++) {
      for (j = 1; j < ny - 1; j++) {
        // Calculate neighbor count
        int nn = currWorld[i + 1][j] + currWorld[i - 1][j] + currWorld[i][j + 1] + currWorld[i][j - 1] +
                 currWorld[i + 1][j + 1] + currWorld[i - 1][j - 1] + currWorld[i - 1][j + 1] + currWorld[i + 1][j - 1];
        // If alive: check if you die, if dead: check if you can produce.
        nextWorld[i][j] = currWorld[i][j] ? (nn == 2 || nn == 3) : (nn == 3);
        // Update population (CRITICAL)
        sum += nextWorld[i][j];
      }
    }
#pragma omp single nowait
    population = sum;
  }

  /* Pointer Swap : nextWorld <-> currWorld */
  tmpWorld = nextWorld;
  nextWorld = currWorld;
  currWorld = tmpWorld;

  // now inside the loop we will print the world, while calculating the next one
  // is it possible to perhaps combine these two parallel regions into one, while also including this iterative loop
  // inside?
  for (iter = 1; iter < maxiter && population; ++iter) {
    /* Use currWorld to compute the updates and store it in nextWorld */
    population = 0;  // do this outside and at the end (instead of start)
    sum = 0;
    proceedToSwap = 0;  // flag variable

#pragma omp parallel num_threads(2) if (numthreads > 1)
    {
#pragma omp single
      {
// task 1: plotting
#pragma omp task private(i, j)
        {
          if (plotter) plotter->plot(iter, population, currWorld);
        }

// task 2: computing
#pragma omp task
        {
#pragma omp parallel num_threads(numthreads - 1) if (numthreads > 2)  // nested parallel enable
          {
#pragma omp for reduction(+: sum) private(j) // we could use collapse(2) but then it would be 2D decomposition, so we go for only one loop parallelization.
            for (i = 1; i < nx - 1; i++) {
              for (j = 1; j < ny - 1; j++) {
                //  calculate neighbor count
                int nn = currWorld[i + 1][j] + currWorld[i - 1][j] + currWorld[i][j + 1] + currWorld[i][j - 1] +
                         currWorld[i + 1][j + 1] + currWorld[i - 1][j - 1] + currWorld[i - 1][j + 1] +
                         currWorld[i + 1][j - 1];
                // if alive check if you die, if dead check if you can produce.
                nextWorld[i][j] = currWorld[i][j] ? (nn == 2 || nn == 3) : (nn == 3);
                // update population (CRITICAL)
                sum += nextWorld[i][j];
              }
            }

#pragma omp single nowait
            {
              // printf("Thread %d reporting in to swap.\n",omp_get_thread_num()); // debug purposes
              population += sum;
            }
          }
        }  // end of compute task

      }  // end of single
    }    // end of 2 threads

    // pointer swap
    tmpWorld = nextWorld;
    nextWorld = currWorld;
    currWorld = tmpWorld;
  }

  // We have print one more, because this was calculated at the last iteration
  if (plotter) plotter->plot(iter, population, currWorld);
}

int populateWorld(params_t params, char **world) {
  int population = 0;
  if (params.game == DEFAULT) {
    // randomly generated
    for (int i = 1; i < params.nx - 1; i++) {
      for (int j = 1; j < params.ny - 1; j++) {
        if (real_rand() < params.prob) {
          world[i][j] = 1;
          population++;
        } else {
          world[i][j] = 0;
        }
      }
    }

  } else if (params.game == BLOCK) {
    // still block-life
    printf("2x2 Block, still life\n");
    int nx2 = params.nx / 2;
    int ny2 = params.ny / 2;
    world[nx2 + 1][ny2 + 1] = world[nx2][ny2 + 1] = world[nx2 + 1][ny2] = world[nx2][ny2] = 1;
    population = 4;
  }
  return population;
}

void game_of_life_start(const params_t params) {
  seed_rand(params.seedVal);
  int i;

  // allocate current world (which you read from)
  char **currWorld = (char **)malloc(sizeof(char *) * params.nx + sizeof(char) * params.nx * params.ny);
  for (i = 0; i < params.nx; i++) currWorld[i] = (char *)(currWorld + params.nx) + i * params.ny;

  // allocate next world (which you write to)
  char **nextWorld = (char **)malloc(sizeof(char *) * params.nx + sizeof(char) * params.nx * params.ny);
  for (i = 0; i < params.nx; i++) nextWorld[i] = (char *)(nextWorld + params.nx) + i * params.ny;

  // reset boundaries
  for (i = 0; i < params.nx; i++) {
    currWorld[i][0] = 0;
    currWorld[i][params.ny - 1] = 0;
    nextWorld[i][0] = 0;
    nextWorld[i][params.ny - 1] = 0;
  }
  for (i = 0; i < params.ny; i++) {
    currWorld[0][i] = 0;
    currWorld[params.nx - 1][i] = 0;
    nextWorld[0][i] = 0;
    nextWorld[params.nx - 1][i] = 0;
  }

  // plot the starting world
  if (!params.disableDisplay) {
    plotter = new GOF_Plotter(params.nx, params.ny);
    plotter->plot(0, 0, currWorld);
  }

  // run
  float ms = 0;
  if (params.numthreads == 1) {
    printf("Serial\n\tProbability: %f\n\tThreads: %d\n\tIterations: %d\n\tProblem Size: %d x %d\n", params.prob,
           params.numthreads, params.maxiter, params.nx, params.ny);

    START_HOST_TIMERS();
    game_of_life_serial();
    STOP_HOST_TIMERS(ms);
  } else {
    printf("Parallel\n\tProbability: %f\n\tThreads: %d\n\tIterations: %d\n\tProblem Size: %d x %d\n", params.prob,
           params.numthreads, params.maxiter, params.nx, params.ny);

    START_HOST_TIMERS();
    game_of_life_parallel();
    STOP_HOST_TIMERS(ms);
  }

  printf("Running time for the iterations: %f sec.\n", ms);

  // frees
  if (!params.disableDisplay) delete plotter;
  free(nextWorld);
  free(currWorld);
}

// TODO: make this whole thing into a class