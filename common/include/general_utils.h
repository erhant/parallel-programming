#pragma once
#include <stdio.h>
#include <string.h>

#ifndef MATCH_INPUT
// @note assumes existence of argv, ac and s variables
#define MATCH_INPUT(s) (!strcmp(argv[ac], (s)))
#endif

void press_any_key();