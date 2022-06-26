#include "gnu_plot.hpp"

GNU_Plotter::GNU_Plotter() { gnu = popen("gnuplot", "w"); }

GNU_Plotter::~GNU_Plotter() { pclose(gnu); }

void GNU_Plotter::setDimensions(int nx, int ny, int offset) {
  m = nx;
  n = ny;
  o = offset;
}

void GNU_Plotter::plotMesh(const char* title, char** mesh) {
   fprintf(gnu, "set title %s\n", title);
  fprintf(gnu, "set size square\n");
  fprintf(gnu, "set key off\n");
  fprintf(gnu, "plot [0:%d] [0:%d] \"-\"\n", m - 1, n - 1);
  for (int i = o; i < m - o; i++) {
    for (int j = o; j < n - o; j++)
      if (mesh[i][j]) {
        fprintf(gnu, "%d %d\n", j, n - i - o);
      }
  }

  fprintf(gnu, "e\n");
  fflush(gnu);
}
