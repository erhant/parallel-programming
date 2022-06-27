#pragma once

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "general_utils.hpp"
#include "plotter.hpp"

extern "C" {
#include "real_rand.h"
}

typedef enum GameType { DEFAULT, BLOCK } game_e;

class GameOfLife : public Project {
 private:
  // parameters
  int nx;              // num. squares in X axis
  int ny;              // num. squares in Y axis
  int numthreads;      // number of threads
  int maxiter;         // max. no of iterations
  float prob;          // probability of placing a cell in world generation
  bool isPlotEnabled;  // enable GNU plotting
  game_e game;         // game type, such as random / glider etc.
  runtype_e runType;   // running type for the application

  // game data
  GameOfLifePlotter *plotter = NULL;  // GNU plotting API
  int population = 0;                 // population of the current world
  bool **currWorld = NULL;            // current world
  bool **nextWorld = NULL;            // next world
  bool **tmpWorld = NULL;             // temporary pointer for swapping
  float runtimeMS = 0;

  // auxillary function for world population
  int populateCurrentWorld();

 public:
  GameOfLife(int nx, int ny, int numthreads, int maxiter, float prob, bool isPlotEnabled, game_e game,
             runtype_e runType);
  void serial();
  void parallel();
};