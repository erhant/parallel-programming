#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "game-of-life.hpp"

int main(int argc, char **argv) {
  // set defaults
  int maxiter = 200;
  int nx = 100;
  int ny = 100;
  float prob = 0.5;
  long seed = 0;
  game_e game = DEFAULT;
  runtype_e runType = SERIAL;
  int numthreads = 1;
  bool isPlotEnabled = false;

  // overrides
  for (int ac = 1; ac < argc; ac++) {
    if (MATCH_ARG("-nx")) {
      nx = atoi(argv[++ac]);
    } else if (MATCH_ARG("-ny")) {
      ny = atoi(argv[++ac]);
    } else if (MATCH_ARG("--iters")) {
      maxiter = atoi(argv[++ac]);
    } else if (MATCH_ARG("--threads")) {
      numthreads = atof(argv[++ac]);
    } else if (MATCH_ARG("--prob")) {
      prob = atof(argv[++ac]);
    } else if (MATCH_ARG("--seed")) {
      seed = atof(argv[++ac]);
    } else if (MATCH_ARG("--plot")) {
      isPlotEnabled = true;
    } else if (MATCH_ARG("--parallel")) {
      runType = PARALLEL;
    } else if (MATCH_ARG("--game")) {
      int gameChoice = atoi(argv[++ac]);
      switch (gameChoice) {
        case 0:
          game = DEFAULT;
          break;
        case 1:
          game = BLOCK;
          break;
        default:
          game = DEFAULT;
      }
    } else {
      printf(
          "Usage: %s "
          "[-nx <points>] "
          "[-ny <points>] "
          "[--iters <iterations>] "
          "[--seed <seed>] "
          "[--prob prob] "
          "[--threads numthreads] "
          "[--game <game no>] "
          "[--parallel] "
          "[--plot]\n",
          argv[0]);
      return (-1);
    }
  }

  // seed randomizer
  seed_rand(seed);

  GameOfLife app(nx, ny, numthreads, maxiter, prob, isPlotEnabled, game, runType);

  return 0;
}
