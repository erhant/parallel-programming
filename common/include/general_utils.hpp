#pragma once

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

class Stopwatch {
 private:
  double startTime;
  double stopTime;
  double getTime();

 public:
  void start();
  double stop();
};

// A project interface
class Project {
 public:
  virtual void serial() = 0;
  virtual void parallel() = 0;
};

void press_any_key();