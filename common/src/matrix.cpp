#include "matrix.hpp"

void free_COO(COO_Matrix *mat) {
  free(mat->rows);
  free(mat->cols);
  free(mat->vals);
  free(mat);
}

COO_Matrix *create_COO(int rows, int cols, int nonzeros) {
  COO_Matrix *mat = (COO_Matrix *)malloc(sizeof(COO_Matrix));
  mat->rows = (int *)calloc(nonzeros, sizeof(int));
  mat->cols = (int *)calloc(nonzeros, sizeof(int));
  mat->vals = (double *)calloc(nonzeros, sizeof(double));
  mat->M = rows;
  mat->N = cols;
  mat->nz = nonzeros;
  mat->isSymmetric = false;  // by default
  if (mat->rows == NULL || mat->cols == NULL || mat->vals == NULL) {
    if (mat->rows) free(mat->rows);
    if (mat->cols) free(mat->cols);
    if (mat->vals) free(mat->vals);
    free(mat);
    return NULL;
  } else
    return mat;
}

COO_Matrix *read_COO(const char *path) {
  MM_typecode matcode;
  FILE *f;
  int i;

  if ((f = fopen(path, "r")) == NULL) {
    return NULL;
  }
  if (mm_read_banner(f, &matcode) != 0) {
    return NULL;
  }

  // Allow only certain matrices
  if (!((mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode))  // has to be real/int/binary
        && mm_is_coordinate(matcode)                                               // has to be in COO format
        && mm_is_sparse(matcode)                                                   // has to be sparse
        && !mm_is_dense(matcode)                                                   // can not be an array
        )) {
    return NULL;
  }

  // Obtain size info
  int M, N, nz;
  if ((mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0) {
    return NULL;
  }

  int *rIndex = (int *)malloc(nz * sizeof(int));
  int *cIndex = (int *)malloc(nz * sizeof(int));
  double *val = (double *)malloc(nz * sizeof(double));

  /* When reading in floats, ANSI C requires the use of the "l"       */
  /* specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /* (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)           */
  /* also use %lg for reading int too */
  if (mm_is_real(matcode) || mm_is_integer(matcode)) {
    double tmp;
    for (i = 0; i < nz; i++) {
      fscanf(f, "%d %d %lg\n", &(rIndex[i]), &(cIndex[i]), &tmp);
      rIndex[i]--;  // 1-indexed to 0-indexed
      cIndex[i]--;  // 1-indexed to 0-indexed
      val[i] = tmp;
    }
  } else if (mm_is_pattern(matcode)) {
    for (i = 0; i < nz; i++) {
      fscanf(f, "%d %d\n", &(rIndex[i]), &(cIndex[i]));
      rIndex[i]--;  // 1-indexed to 0-indexed
      cIndex[i]--;  // 1-indexed to 0-indexed
      val[i] = 1.0;
    }
  } else
    return NULL;

  if (f != stdin) fclose(f);
  COO_Matrix *mat = (COO_Matrix *)malloc(sizeof(COO_Matrix));
  mat->M = M;
  mat->N = N;
  mat->nz = nz;
  mat->rows = rIndex;
  mat->cols = cIndex;
  mat->vals = val;
  mat->isSymmetric = mm_is_symmetric(matcode);
  if (mm_is_real(matcode)) mat->type = 'r';
  if (mm_is_integer(matcode)) mat->type = 'i';
  if (mm_is_pattern(matcode)) mat->type = 'p';
  return mat;
}

void duplicate_off_diagonals(COO_Matrix *mat) {
  // count number of off-diagonal entries
  int off_diagonals = 0, i;
  for (i = 0; i < mat->nz; i++)
    if (mat->rows[i] != mat->cols[i]) off_diagonals++;

  // allocate new memory for the actual matrix
  int true_nz = mat->nz + off_diagonals;  // 2 * off_diagonals + (nz - off_diagonals)
  int *new_rows = (int *)malloc(true_nz * sizeof(int));
  int *new_cols = (int *)malloc(true_nz * sizeof(int));
  double *new_vals = (double *)malloc(true_nz * sizeof(double));

  // populate the new values
  int new_i = 0;
  for (i = 0; i < mat->nz; i++) {
    // copy original
    new_rows[new_i] = mat->rows[i];
    new_cols[new_i] = mat->cols[i];
    new_vals[new_i] = mat->vals[i];
    new_i++;
    // if off diagonal, copy the symmetric value
    if (mat->rows[i] != mat->cols[i]) {
      new_cols[new_i] = mat->rows[i];  // row to col
      new_rows[new_i] = mat->cols[i];  // col to row
      new_vals[new_i] = mat->vals[i];  // but same val
      new_i++;
    }
  }

  // free old pointers
  free(mat->rows);
  free(mat->cols);
  free(mat->vals);
  // assign new pointers
  mat->rows = new_rows;
  mat->cols = new_cols;
  mat->vals = new_vals;
  // now the matrix is not symmetric (values are explicit)
  mat->isSymmetric = false;
  // update nz value
  mat->nz = true_nz;
  assert(new_i == true_nz);
}

CSR_Matrix<double> *COO_to_CSR(COO_Matrix *coo) {
  CSR_Matrix<double> *csr = create_CSR<double>(coo->M, coo->N, coo->nz);
  if (!csr) return NULL;

  int i;
  for (i = 0; i < coo->nz; i++) csr->rowptr[coo->rows[i] + 2]++;
  for (i = 2; i < coo->M + 2; i++) csr->rowptr[i] += csr->rowptr[i - 1];
  for (i = 0; i < coo->nz; i++) {
    csr->cols[csr->rowptr[coo->rows[i] + 1]] = coo->cols[i];
    csr->vals[csr->rowptr[coo->rows[i] + 1]++] = coo->vals[i];
  }
  assert(csr->rowptr[csr->M] == coo->nz);
  return csr;
}
