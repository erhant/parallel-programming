#include "real_rand.h"

int seed_rand(long sd) {
  static int seed_me = 0; /* "...Seymour, seed me all night long..." */
  if (sd) {
    seed_me = sd;
  } else {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    long hashed = ((tp.tv_sec & ~tp.tv_usec) | (~tp.tv_sec & tp.tv_usec));
    seed_me = hashed;
  }
  srand48(seed_me);
  return (seed_me);
}

double real_rand() { return drand48(); }
