#include "matrix.hpp"

/**
 * @brief Free CSR matrix from the host memory.
 *
 * @tparam T type of the values in matrix
 * @param mat CSR matrix
 */
template <typename T>
void free_CSR(CSR_Matrix<T> *mat) {
  free(mat->rowptr);
  free(mat->cols);
  free(mat->vals);
  free(mat);
}

/**
 * @brief Allocate memory for a CSR matrix in host memory.
 *
 * @param rows number of rows
 * @param cols number of columns
 * @param nonzeros number of non-zero values
 */
template <typename T>
CSR_Matrix<T> *create_CSR(int rows, int cols, int nonzeros) {
  CSR_Matrix<T> *mat = (CSR_Matrix<T> *)malloc(sizeof(CSR_Matrix<T>));
  mat->rowptr = (int *)calloc((rows + 2), sizeof(int));
  mat->cols = (int *)calloc(nonzeros, sizeof(int));
  mat->vals = (T *)calloc(nonzeros, sizeof(T));
  mat->M = rows;
  mat->N = cols;
  mat->nz = nonzeros;
  if (mat->rowptr == NULL || mat->cols == NULL || mat->vals == NULL) {
    if (mat->rowptr) free(mat->rowptr);
    if (mat->cols) free(mat->cols);
    if (mat->vals) free(mat->vals);
    free(mat);
    return NULL;
  } else
    return mat;
}

/**
 * @brief Make a copy of the given matrix, with type-casting.
 *
 * @tparam TSRC source type
 * @tparam TDEST destination type
 * @param mat source CSR matrix
 */
template <typename TSRC, typename TDEST>
CSR_Matrix<TDEST> *duplicate_CSR(CSR_Matrix<TSRC> *mat) {
  CSR_Matrix<TDEST> *matnew = create_CSR<TDEST>(mat->M, mat->N, mat->nz);
  if (!matnew) return NULL;

  for (int i = 0; i < mat->M; i++) {
    matnew->rowptr[i] = mat->rowptr[i];
    for (int j = mat->rowptr[i]; j < mat->rowptr[i + 1]; j++) {
      matnew->cols[j] = mat->cols[j];
      matnew->vals[j] = (TDEST)mat->vals[j];
    }
  }
  matnew->rowptr[matnew->M] = mat->rowptr[mat->M];
  return matnew;
}

/**
 * @brief Free ELLPACK-R matrix from the host memory.
 *
 * @tparam T type of the values in matrix
 * @param mat ELLPACK-R matrix
 */
template <typename T>
void free_ELLR(ELLR_Matrix<T> *mat) {
  free(mat->rowlen);
  free(mat->cols);
  free(mat->vals);
  free(mat);
}
