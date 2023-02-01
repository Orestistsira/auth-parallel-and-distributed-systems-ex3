#include <iostream>
#include <fstream>
#include <cassert>
#include <limits>
#include <string.h>
#include <cusparse.h>         // cusparseSpVV

#include "fglt.hpp"

void readMTX
(
 mwIndex       **       row,
 mwIndex       **       col,
 mwSize         * const n,
 mwSize         * const m,
 char    const  * const filename
){

  // ~~~~~~~~~~ variable declarations
  mwIndex *row_coo, *col_coo;
  mwSize  n_col, m_mat;
  char mmx[20], b1[20], b2[20], b3[20], b4[20];
  bool issymmetric = false;
  
  // ~~~~~~~~~~ read matrix
  
  // open the file
  std::ifstream fin( filename );

  // check if file exists
  if( fin.fail() ){
    std::cerr << "File " << filename << " could not be opened! Aborting..." << std::endl;
    exit(1);
  }

  // read banner
  fin >> mmx >> b1 >> b2 >> b3 >> b4;

  // parse banner
  if ( strcmp( b1, "matrix" ) ){
    std::cerr << "Currently works only with 'matrix' option, aborting..." << std::endl;
    exit(1);
  }

  if ( strcmp( b2, "coordinate" ) ){
    std::cerr << "Currently works only with 'coordinate' option, aborting..." << std::endl;
    exit(1);
  }
  
  if ( strcmp( b3, "pattern" ) ){
    std::cerr << "Currently works only with 'pattern' format, aborting..." << std::endl;
    exit(1);
  }

  if ( !strcmp( b4, "symmetric" ) ){
    issymmetric = true;
  }



  // ignore headers and comments
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  while (fin.peek() == '%') fin.ignore(2048, '\n');

  // read defining parameters
  fin >> n[0] >> n_col >> m[0];

  if (issymmetric) m_mat = 2*m[0];
  else             m_mat = m[0];
  
  assert( n[0] == n_col );

  // allocate space for COO format
  row_coo = static_cast<mwIndex *>( malloc(m_mat * sizeof(mwIndex)) );
  col_coo = static_cast<mwIndex *>( malloc(m_mat * sizeof(mwIndex)) );
  
  // read the COO data
  mwIndex k = 0;
  for (mwIndex l = 0; l < m[0]; l++){
    fin >> row_coo[k] >> col_coo[k];
    if (issymmetric)
      if (row_coo[k] == col_coo[k])  m_mat -= 2; // we do not keep self-loop, remove edges
      else { row_coo[k+1] = col_coo[k]; col_coo[k+1] = row_coo[k]; k += 2; } // put symmetric edge
    else
      if (row_coo[k] == col_coo[k])  m_mat -= 1; // we do not keep self-loop, remove edge
      else k++;
  }

  m[0] = m_mat;
  row_coo =static_cast<mwIndex *>( realloc( row_coo, m[0] * sizeof(mwIndex) ) );
  col_coo =static_cast<mwIndex *>( realloc( col_coo, m[0] * sizeof(mwIndex) ) );

  // close connection to file
  fin.close();

  // ~~~~~~~~~~ transform COO to CSC
  row[0] = static_cast<mwIndex *>( malloc( m[0]   * sizeof(mwIndex)) );
  col[0] = static_cast<mwIndex *>( calloc( (n[0]+1),  sizeof(mwIndex)) );

  // ----- find the correct column sizes
  for (mwSize l = 0; l < m[0]; l++){            
    col[0][ col_coo[l]-1 ]++;
  }
  
  for(mwSize i = 0, cumsum = 0; i < n[0]; i++){     
    int temp = col[0][i];
    col[0][i] = cumsum;
    cumsum += temp;
  }
  col[0][n[0]] = m[0];
  
  // ----- copy the row indices to the correct place
  for (mwSize l = 0; l < m[0]; l++){
    int col_l = col_coo[l]-1;
    int dst = col[0][col_l];
    row[0][dst] = row_coo[l]-1;
    
    col[0][ col_l ]++;
  }
  
  // ----- revert the column pointers
  for(mwSize i = 0, last = 0; i < n[0]; i++) {     
    int temp = col[0][i];
    col[0][i] = last;

    last = temp;
  }

  // ~~~~~~~~~~ deallocate memory
  free( row_coo );
  free( col_coo );

}

void compute_d4(mwIndex *row, mwIndex *col, mwSize n, mwSize m, double *d4){

    double *fl = (double *) calloc(n, sizeof(double));
    int *pos = (int *) calloc(n, sizeof(int));
    int *isNgbh = (int *) calloc(n, sizeof(int));
    mwIndex *isUsed = (mwIndex *) calloc(n, sizeof(mwIndex));

    for(int i=0;i<n;i++){
    // setup the count of nonzero columns (j) visited for this row (i)
        mwIndex cnt = 0;

        // --- loop through every nonzero element A(i,k)
        for (mwIndex id_i = col[i]; id_i < col[i+1]; id_i++){

            // get the column (k)
            mwIndex k = row[id_i];

            isNgbh[k] = id_i+1;
            
            // --- loop through all nonzero elemnts A(k,j)
            for (mwIndex id_k = col[k]; id_k < col[k+1]; id_k++){

                // get the column (j)
                mwIndex j = row[id_k];

                if (i == j) continue;

                // if this column is not visited yet for this row (i), then set it
                if (!isUsed[j]) {
                    fl[j]      = 0;  // initialize corresponding element
                    isUsed[j]  = 1;  // set column as visited
                    pos[cnt++] = j;  // add column position to list of visited
                }

                // increase count of A(i,j)
                fl[j]++;
                
            }

        }

        // --- perform reduction on [cnt] non-empty columns (j) 
        for (mwIndex l=0; l<cnt; l++) {

            // get next column number (j)
            mwIndex j = pos[l];

            if (isNgbh[j]) {
                    
                d4[i]  += fl[j];
            }
            
            // declare it non-used
            isUsed[j] = 0;
        }

        d4[i]  /= 2;

        for (mwIndex id_i = col[i]; id_i < col[i+1]; id_i++){

          // get the column (k)
          mwIndex k = row[id_i];

          isNgbh[k] = 0;
        }

    }
}

void computeRaw(mwIndex *row, mwIndex *col, mwSize n, mwSize m, double **d){
  //d0, d1
  for(mwSize i=0;i<n;i++){
    d[0][i] = 1;
    d[1][i] = col[i+1] - col[i];
    d[3][i] = d[1][i] * (d[1][i] - 1) * 0.5;
  }

  //d2, d3
  for(mwSize i=0;i<n;i++){
    for(mwIndex id_i = col[i]; id_i < col[i+1]; id_i++){

      // get the column (k)
      mwIndex k = row[id_i];
        
      // --- matrix-vector products
      d[2][i] += d[1][k];
    }

    d[2][i] -= d[1][i];
  }

  compute_d4(row, col, n, m, d[4]);

  FILE* f;

  f = fopen("raw_results.txt", "w");
  for(mwSize i=0;i<n;i++){
    d[2][i] = d[2][i] -  2 * d[4][i];
    d[3][i] -= d[4][i];
    fprintf(f, "%.1f %.1f %.1f %.1f, %.1f\n", d[0][i], d[1][i], d[2][i], d[3][i], d[4][i]);
  }

  fclose(f);
}

#define BLOCK_SIZE 512

__global__ void compute_d0_d1_d3_Kernel(mwIndex *col, double *d0, double *d1, double *d3, mwSize n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n){
    d0[i] = 1;
    d1[i] = col[i+1] - col[i];
    d3[i] = d1[i] * (d1[i] - 1) * 0.5;
  }
}

__global__ void compute_d2_Kernel(mwIndex *row, mwIndex *col, double *d1, double *d2, mwSize n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n){
    double sum = 0;
    for(mwIndex id_i = col[i]; id_i < col[i+1]; id_i++){

      // get the column (k)
      mwIndex k = row[id_i];
        
      // --- matrix-vector products
      sum += d1[k];
    }
    d2[i] = sum - d1[i];
  }
}

// __global__ void sparseMatrixSquareKernel(const int *row, const int *col, const float *valA, float *valC, int *colC, int *rowC, int N) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;

//   if (i < N) {
//     int rowStart = col[i];
//     int rowEnd = col[i + 1];

//     int nnzC = 0;
//     for (int i = rowStart; i < rowEnd; ++i) {
//       int colA_i = row[i];
//       float valA_i = valA[i];

//       int colStart = col[colA_i];
//       int colEnd = col[colA_i + 1];

//       for (int j = colStart; j < colEnd; ++j) {
//         int colA_j = row[j];
//         float valA_j = valA[j];

//         int colStart2 = col[colA_j];
//         int colEnd2 = col[colA_j + 1];

//         for (int k = colStart2; k < colEnd2; ++k) {
//           int colA_k = row[k];

//           if (colA_k == i) {
//             float val = valA_i * valA_j;
//             if (val != 0.0f) {
//               valC[nnzC] = val;
//               rowC[nnzC] = colA_j;
//               ++nnzC;
//             }
//           }
//         }
//       }
//       colC[i + 1] = nnzC;
//     }
//   }
// }

__global__ void square_csc_sparse_matrix(mwSize n, mwSize m, mwIndex *A_row_idx, mwIndex *A_col_ptr, unsigned int *result_row_idx, unsigned int *result_col_ptr) {
  
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (tid < m) {
    int row = A_row_idx[tid];
    //int col = A_col_ptr[row];

    for (int i = A_col_ptr[row]; i < A_col_ptr[row + 1]; i++) {
      int j = A_row_idx[i];
      int new_col = result_col_ptr[j];

      for (int k = A_col_ptr[j]; k < A_col_ptr[j + 1]; k++) {
        int l = A_row_idx[k];
        if (l == row) {
          result_row_idx[new_col + atomicAdd(&result_col_ptr[j + 1], 1) - 1] = row;
        }
      }
    }
  }
}

__global__ void compute_d4_Kernel(unsigned int *col, double *d4, mwSize n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n){
    d4[i] = col[i+1] - col[i];
  }
}


void cudaComputeRaw(mwIndex *row, mwIndex *col, mwSize n, mwSize m, double **d){
  double *d0, *d1, *d2, *d3, *d4;

  d0 = d[0];
  d1 = d[1];
  d2 = d[2];
  d3 = d[3];
  d4 = d[4];

  double *d0D, *d1D, *d2D, *d3D, *d4D;

  //d0, d1, d3------------------------------------

  cudaMalloc((void**)&d0D, n * sizeof(double));
  cudaMalloc((void**)&d1D, n * sizeof(double));
  cudaMalloc((void**)&d3D, n * sizeof(double));

  mwIndex *colD;
  cudaMalloc((void**)&colD, (n+1) * sizeof(mwIndex));
  cudaMemcpy(colD, col, (n+1) * sizeof(mwIndex), cudaMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_SIZE); //thread per block
  dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE); //num of blocks
  compute_d0_d1_d3_Kernel<<<dimGrid, dimBlock>>>(colD, d0D, d1D, d3D, n);

  cudaMemcpy(d0, d0D, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(d1, d1D, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(d3, d3D, n * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d0D);
  cudaFree(d3D);

  //d2------------------------------------

  cudaMalloc((void**)&d2D, n * sizeof(double));
  
  mwIndex *rowD;
  cudaMalloc((void**)&rowD, m * sizeof(double));

  cudaMemcpy(rowD, row, m * sizeof(double), cudaMemcpyHostToDevice);

  compute_d2_Kernel<<<dimGrid, dimBlock>>>(rowD, colD, d1D, d2D, n);

  cudaMemcpy(d2, d2D, n * sizeof(double), cudaMemcpyDeviceToHost);
  
  //cudaDeviceSynchronize();

  //d4
  // unsigned int *resRow, *resCol;

  // cudaMalloc((void**)&resCol, (n+1) * sizeof(unsigned int));
  // cudaMalloc((void**)&resRow, m * sizeof(unsigned int));
  // cudaMemset(resCol, 0, (n+1) * sizeof(unsigned int));
  // cudaMemset(resRow, 0, m * sizeof(unsigned int));

  // cudaMalloc((void**)&d4D, n * sizeof(double));
  
  // dim3 dimBlock2(BLOCK_SIZE); //thread per block
  // dim3 dimGrid2((m + BLOCK_SIZE - 1) / BLOCK_SIZE); //num of blocks
  // square_csc_sparse_matrix<<<dimGrid2, dimBlock2>>>(n, m, rowD, colD, resRow, resCol);

  // compute_d4_Kernel<<<dimGrid, dimBlock>>>(resCol, d4D, n);

  // cudaMemcpy(d4, d4D, n * sizeof(double), cudaMemcpyDeviceToHost);

  // cudaFree(d1D);
  // cudaFree(d2D);
  // cudaFree(d4D);
  // cudaFree(colD);
  // cudaFree(rowD);
  // cudaFree(resCol);
  // cudaFree(resRow);

  //d2 cuSparse--------------------------------------------
  
  // mwIndex *rowD;
  // double *valuesD;
  // double *values = (double*) malloc(m * sizeof(double));
  // for(int i=0;i<m;i++){
  //   values[i] = 1;
  // }

  // cudaMalloc((void**)&rowD, m * sizeof(mwIndex));
  // cudaMalloc((void**)&valuesD, m * sizeof(double));
  // cudaMemcpy(rowD, row, m * sizeof(mwIndex), cudaMemcpyHostToDevice);
  // cudaMemcpy(valuesD, values, m * sizeof(double), cudaMemcpyHostToDevice);

  // cudaMalloc((void**)&d2D, n * sizeof(double));
  // cudaMemcpy(d2D, d1D, n * sizeof(double), cudaMemcpyDeviceToDevice);


  // double     alpha           = 1.0;
  // double     beta            = -1.0;

  // cusparseHandle_t     handle = NULL;
  // cusparseSpMatDescr_t matA;
  // cusparseDnVecDescr_t vecX, vecY;
  // void*                dBuffer    = NULL;
  // size_t               bufferSize = 0;

  // cusparseCreate(&handle);

  // cusparseCreateCsc(&matA, n, n, m, colD, rowD, valuesD, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

  // cusparseCreateDnVec(&vecX, n, d1D, CUDA_R_64F);
  // cusparseCreateDnVec(&vecY, n, d2D, CUDA_R_64F);

  // cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  // cudaMalloc(&dBuffer, bufferSize);

  // cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

  // cusparseDestroySpMat(matA);
  // cusparseDestroyDnVec(vecX);
  // cusparseDestroyDnVec(vecY);
  // cusparseDestroy(handle);

  // cudaMemcpy(d2, d2D, n * sizeof(double), cudaMemcpyDeviceToHost);

  // cudaFree(d1D);
  // cudaFree(d2D);
  // cudaFree(colD);
  // cudaFree(rowD);

  //--------------------------------------------

  FILE* f;
  f = fopen("raw_results_cuda.txt", "w");

  for(mwSize i = 0; i < n; i++){
    fprintf(f, "%.1f %.1f %.1f %.1f %.1f\n", d0[i], d1[i], d2[i], d3[i], d4[i]);
  }
  fclose(f);

}

void printArray(mwIndex* array, mwSize n){
  for(int i=0;i<n;i++){
      printf("%ld\n", array[i]);
  }
  printf("\n");
}

int main(int argc, char **argv)
{

  // ~~~~~~~~~~ variable declarations
  mwIndex *row, *col;
  mwSize  n, m;
  std::string filename = "graph.mtx";

  // ~~~~~~~~~~ parse inputs

  // ----- retrieve the (non-option) argument:
  if ( (argc <= 1) || (argv[argc-1] == NULL) || (argv[argc-1][0] == '-') ) {
    // there is NO input...
    std::cout << "No filename provided, using 'graph.mtx'." << std::endl;
  }
  else {
    // there is an input...
    filename = argv[argc-1];
    std::cout << "Using graph stored in '" << filename << "'." << std::endl;
  }
  
  readMTX( &row, &col, &n, &m, filename.c_str() );

  printf("number of columns = %ld\n", n);
  printf("non zero elements = %ld\n", m);
  //printArray(col, n + 1);

  //printArray(row, m);
  int graphletSize = 5;
  double **d  = (double **) malloc(graphletSize * sizeof(double *));

  for (int igraph = 0; igraph < graphletSize; igraph++){
    d[igraph] = (double *) calloc(n, sizeof(double));
  }

  struct timeval startwtime, endwtime;
  double duration;

  gettimeofday (&startwtime, NULL);
  computeRaw(row, col, n, m, d);
  gettimeofday (&endwtime, NULL);
  duration = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
  printf("[computeRaw took %.4f seconds]\n", duration);
  

  gettimeofday (&startwtime, NULL);
  cudaComputeRaw(row, col, n, m, d);
  gettimeofday (&endwtime, NULL);
  duration = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
  printf("[cudaComputeRaw took %.4f seconds]\n", duration);

  for (int igraph = 0; igraph < graphletSize; igraph++){
    free(d[igraph]);
  }
  free(d);
  free( row );
  free( col );
  
}