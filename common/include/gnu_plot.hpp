#pragma once

#include <stdio.h>

class GNUPlotter {
 private:
  FILE* gnu = NULL;
  int m;  // x-dimension size
  int n;  // y-dimension size
  int o;  // offset for both x and y (for ghost boundaries)

 public:
  // Constructor opens GNU pipe in 'w' mode
  GNUPlotter();

  // Destructor closes GNU pipe
  ~GNUPlotter();

  // Set the dimensions of plot, instead of providing it on each plot
  void setDimensions(int nx, int ny, int offset);

  void plotMesh(const char* title, bool** mesh);
};