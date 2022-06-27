#include "general_utils.hpp"

void press_any_key(const char* msg = "Press any key to continue.") {
  printf("%s\n", msg);
  getchar();
}

double Stopwatch::getTime() {
  struct timeval TV;
  struct timezone TZ;
  const int RC = gettimeofday(&TV, &TZ);
  if (RC == -1) {
    printf("ERROR: Bad call to gettimeofday\n");
    return (-1);
  }
  return (((double)TV.tv_sec) + 1.0e-6 * ((double)TV.tv_usec));
}

void Stopwatch::start() {
  if (!isRunning) {
    this->startTime = this->getTime();
    this->isRunning = true;
  }
}

double Stopwatch::stop() {
  if (isRunning) {
    this->stopTime = this->getTime();
    this->isRunning = false;

  } else {
    return 0;
  }
}

double Project::run(runtype_e runType) {
  // print runtime info
  this->printParameters(runType);

  // start stopwatch
  this->sw->start();

  // run the program
  if (runType == SERIAL) {
    this->serial();
  } else if (runType == PARALLEL) {
    this->parallel();
  }

  // stop stopwatch and report elapsed time
  return this->sw->stop();
}

Project::Project() { this->sw = new Stopwatch(); }

Project::~Project() { delete this->sw; }