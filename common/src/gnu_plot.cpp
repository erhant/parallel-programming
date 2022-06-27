#include "gnu_plot.hpp"

GNUPlotter::GNUPlotter() { gnu = popen("gnuplot", "w"); }

GNUPlotter::~GNUPlotter() { pclose(gnu); }

void GNUPlotter::setDimensions(int nx, int ny, int offset) {
  m = nx;
  n = ny;
  o = offset;
}

void GNUPlotter::plotMesh(const char* title, bool** mesh) {
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
