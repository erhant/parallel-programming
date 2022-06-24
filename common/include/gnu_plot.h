#pragma once

#include <stdio.h>

/**
 * Function to plot the 2D array 'gnuplot' is instantiated via a pipe and
 * the values to be plotted are passed through, along with gnuplot commands.
 *
 * @param t iteration
 * @param m
 * @param n
 * @param mesh
 * */
int plot_mesh(int t, int m, int n, char **mesh);