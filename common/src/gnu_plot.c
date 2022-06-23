#include "gnu_plot.h"

FILE *gnu = NULL;

int MeshPlot(int t, int m, int n, char **mesh) {
  int i, j;
  char iter[60];
  sprintf(iter, "\"Iter = %d\"", t);

  if (gnu == NULL) gnu = popen("gnuplot", "w");

  fprintf(gnu, "set title %s\n", iter);
  fprintf(gnu, "set size square\n");
  fprintf(gnu, "set key off\n");
  fprintf(gnu, "plot [0:%d] [0:%d] \"-\"\n", m - 1, n - 1);
  for (i = 1; i < m - 1; i++) {
    for (j = 1; j < n - 1; j++)
      if (mesh[i][j]) {
        fprintf(gnu, "%d %d\n", j, n - i - 1);
      }
  }

  fprintf(gnu, "e\n");
  fflush(gnu);

  return (0);
}
