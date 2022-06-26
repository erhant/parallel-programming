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

void game_of_life_start(const params_t params);

#define GOF_WORLD_UPDATE 0
#define GOF_WORLD_PLOT 1

#include "params.hpp"