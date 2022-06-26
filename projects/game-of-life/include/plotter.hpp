#pragma once

#include "gnu_plot.hpp"

class GOF_Plotter : public GNU_Plotter {
 public:
  GOF_Plotter(int nx, int ny);

  void plot(const int iter, int population, char **mesh);
};
