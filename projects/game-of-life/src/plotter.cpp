#include "plotter.hpp"

GameOfLifePlotter::GameOfLifePlotter(int nx, int ny) { setDimensions(nx, ny, 1); }

void GameOfLifePlotter::plot(const int iter, int population, bool **mesh) {
  char title[60];
  sprintf(title, "\"i: %d - p: %d\"", iter, population);
  plotMesh(title, mesh);
}