#pragma once

#include <stdio.h>

class GNU_Plotter {
 private:
  FILE* gnu = NULL;
  int m;  // x-dimension size
  int n;  // y-dimension size
  int o;  // offset for both x and y (for ghost boundaries)

 public:
  // Constructor opens GNU pipe in 'w' mode
  GNU_Plotter();

  // Destructor closes GNU pipe
  ~GNU_Plotter();

  // Set the dimensions of plot, instead of providing it on each plot
  void setDimensions(int nx, int ny, int offset);

  void plotMesh(const char* title, char** mesh;
};