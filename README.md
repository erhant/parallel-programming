# Parallel Programming

Various parallel programming implementations.

- [Sparse Matrix-Vector Multiplication](./projects/spmv/)
- [Cardiac Simulation](./projects/cardiac-simulation)
- [Jacobi Stencil](./projects/jacobi-1d-stencil/)
- [Jacobi Method](./projects/jacobi-method/)
- [Game of Life](./projects/game-of-life/)
- [Noise Remover](./projects/noise-remover/)
- [$N$-Queens Problem](./projects/nqueens-problem/)
- [Primes](./projects/primes)

Also includes [header files](./common/) for:

- Safe CUDA function and kernel calls, timer macros.
- MatrixMarket reader for COO format with COO $\to$ CSR & CSR conversion
- GNU Plotting
- Random Numbers

Most of the files have the following sturcture: a `main` file that parses command line arguments and calls the actual application. Each application is a class with two functions: `serial` and `parallel` to run the application in either mode respectively.
