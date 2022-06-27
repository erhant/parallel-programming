#pragma once

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// @note assumes existence of argv, ac and s variables
#define MATCH_ARG(s) (!strcmp(argv[ac], (s)))

// Enum for Running type of the project
typedef enum RunningType { SERIAL, PARALLEL } runtype_e;

/**
 * A simple stopwatch.
 */
class Stopwatch {
 private:
  double startTime = 0;
  double stopTime = 0;
  bool isRunning = false;
  double getTime();

 public:
  void start();
  double stop();
};

/**
 * Base class for all projects.
 */
class Project {
 private:
  Stopwatch* sw = NULL;

 protected:
  virtual void serial() = 0;
  virtual void parallel() = 0;
  virtual void printParameters(runtype_e runType) = 0;

 public:
  double run(runtype_e runType);
};

/**
 * A tiny utility function to wait for arbitrary user input.
 * @param msg custom message prompt
 */
void press_any_key(const char* msg);