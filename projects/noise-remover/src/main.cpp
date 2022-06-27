

int main() {
  if (argc < 2) {
    printf("Usage: %s [-i < filename>] [-iter <n_iter>] [-l <lambda>] [-o <outputfilename>]\n", argv[0]);
    return (-1);
  }
  for (int ac = 1; ac < argc; ac++) {
    if (MATCH_ARG("-i")) {
      filename = argv[++ac];
    } else if (MATCH_ARG("-iter")) {
      n_iter = atoi(argv[++ac]);
    } else if (MATCH_ARG("-l")) {
      lambda = atof(argv[++ac]);
    } else if (MATCH_ARG("-o")) {
      outputname = argv[++ac];
    } else {
      printf("Usage: %s [-i < filename>] [-iter <n_iter>] [-l <lambda>] [-o <outputfilename>]\n", argv[0]);
      return (-1);
    }
  }
}