/* Random number generation routines based on system 48-bit LC RNG:
 * Written by Thomas Kammeyer, UCSD
 *
 * double real_rand() automatically seeds itself if need be
 *                    and returns random numbers from [0,1)
 *
 * 05/22/96 jcm Correction to use of gettimeofday
 *
 * Requirements: Be sure and load with -lm
 */

/*
 * This scheme uses gettmeofday() to seed the random number generator,
 * and has obvious flaws.
 *
 */

#pragma once

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

int seed_rand(long sd);

double real_rand();