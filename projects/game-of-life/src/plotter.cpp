#include "plotter.hpp"

GOF_Plotter::GOF_Plotter(int nx, int ny) { setDimensions(nx, ny, 1); }

void GOF_Plotter::plot(const int iter, int population, char **mesh) {
  char title[60];
  sprintf(title, "\"i: %d - p: %d\"", iter, population);
  plotMesh(title, mesh);
}