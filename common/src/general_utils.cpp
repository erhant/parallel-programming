#include "general_utils.hpp"

void press_any_key() {
  printf("Press any key to continue.\n");
  getchar();
}

class Stopwatch {
 private:
  double startTime = 0;
  double stopTime = 0;
  bool isRunning = false;
  double getTime() {
    struct timeval TV;
    struct timezone TZ;
    const int RC = gettimeofday(&TV, &TZ);
    if (RC == -1) {
      printf("ERROR: Bad call to gettimeofday\n");
      return (-1);
    }
    return (((double)TV.tv_sec) + 1.0e-6 * ((double)TV.tv_usec));
  }

 public:
  void start() {
    if (!isRunning) {
      startTime = this->getTime();

      isRunning = true;
    }
  }
  double stop() {
    if (isRunning) {
      isRunning = false;
    } else {
      return 0;
    }
  }
}