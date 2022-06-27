#include "first-primes.hpp"

int main(int argc, char **argv) {
  // default parameters
  unsigned int n = 100;
  runtype_e runType = SERIAL;
  // overrides
  for (int ac = 1; ac < argc; ac++) {
    if (MATCH_ARG("-n")) {
      int x = atoi(argv[++ac]);
      n = x < 0 ? -x : x;  // rather than underflow, just flip it
    } else if (MATCH_ARG("-p")) {
      runType = PARALLEL;
    } else {
      printf(
          "Usage: %s "
          "[-n <number of primes>]"
          "[-p]\n",
          argv[0]);
      return -1;
    }
  }

  FirstPrimes fp(n);
  double time = fp.run(runType);
  printf("Finished in %f ms.", n, time);
}