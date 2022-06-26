#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "game-of-life.hpp"
#include "params.hpp"

int main(int argc, char **argv) {
  int i, j, ac;

  // set default input parameters
  params_t params;
  params.maxiter = 200;
  params.nx = 100;
  params.ny = 100;
  params.prob = 0.5;
  params.seedVal = 0;
  params.game = DEFAULT;
  params.isSingleStep = false;
  params.numthreads = 1;
  params.disableDisplay = false;
  params.disableLogging = false;

  /* Over-ride with command-line input parameters (if any) */
  // ./life -i MAXITER -t NUMTHREAD -p PROB -s SEEDVAL -step SINGLESTEP -g GAMENO -d
  // -d takes no parameters but it will disable display, I set it to be -d by default.
  for (ac = 1; ac < argc; ac++) {
    if (MATCH_ARG("-nx")) {
      params.nx = atoi(argv[++ac]);
    } else if (MATCH_ARG("-ny")) {
      params.ny = atoi(argv[++ac]);
    } else if (MATCH_ARG("-i")) {
      params.maxiter = atoi(argv[++ac]);
    } else if (MATCH_ARG("-t")) {
      params.numthreads = atof(argv[++ac]);
    } else if (MATCH_ARG("-p")) {
      params.prob = atof(argv[++ac]);
    } else if (MATCH_ARG("-s")) {
      params.seedVal = atof(argv[++ac]);
    } else if (MATCH_ARG("-step")) {
      params.isSingleStep = true;
    } else if (MATCH_ARG("-d")) {
      params.disableDisplay = true;
    } else if (MATCH_ARG("-l")) {
      params.disableLogging = true;
    } else if (MATCH_ARG("-g")) {
      int gameChoice = atoi(argv[++ac]);
      switch (gameChoice) {
        case 0:
          params.game = DEFAULT;
        default:
          params.game = DEFAULT;
      }
    } else {
      printf(
          "Usage: %s [-nx <points>] [-ny <points>] [-i <iterations>] [-s <seed>] [-p prob] [-t numthreads] [-step] [-g "
          "<game #>] "
          "[-d]\n",
          argv[0]);
      return (-1);
    }
  }

  // Increment sizes to account for boundary ghost cells
  params.nx = params.nx + 2;
  params.ny = params.ny + 2;

  game_of_life_start(params);

  return (0);
}
