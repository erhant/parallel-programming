#pragma once
#include <stdio.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// @note assumes existence of argv, ac and s variables
#define MATCH_ARG(s) (!strcmp(argv[ac], (s)))

// Start OMP time for stopwatch
#define START_HOST_TIMERS() double start_omptime = omp_get_wtime(), stop_omptime;

// Stop OMP time to measure elapsed time
#define STOP_HOST_TIMERS(ms) \
  stop = omp_get_wtime();    \
  ms = (stop - start) / 1000.0;

void press_any_key();