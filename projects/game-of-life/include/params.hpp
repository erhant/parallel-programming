typedef enum GameType { DEFAULT, BLOCK } game_e;

typedef struct Parameters {
  int nx;               // num. squares in X axis
  int ny;               // num. squares in Y axis
  int numthreads;       // number of threads
  int maxiter;          // max. no of iterations
  int seed;             // randomization seed
  long seedVal;         // ???
  bool isSingleStep;    // enable step-by-step execution
  float prob;           // probability of placing a cell in world generation
  bool disableDisplay;  // enable GNU plotting
  bool disableLogging;  // enable logging
  game_e game;          // game type, such as random / glider etc.
} params_t