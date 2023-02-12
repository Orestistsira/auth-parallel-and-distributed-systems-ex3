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


#define BLOCK_SIZE 512

__global__ void compute_d0_d1_d3_Kernel(mwIndex *col, double *d0, double *d1, double *d3, mwSize n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n){
    d0[i] = 1;
    d1[i] = col[i+1] - col[i];
    d3[i] = d1[i] * (d1[i] - 1) * 0.5;
  }
}

//matrix-vector multiplication
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

__device__ double warp_reduce(double sum){
  unsigned FULL_WARP_MASK = 0xffffffff;

  for(int offset = warpSize / 2; offset > 0; offset /= 2){
    sum += __shfl_down_sync(FULL_WARP_MASK, sum, offset);
  }
  return sum;
}

__global__ void compute_d2_vector_Kernel(mwIndex *row, mwIndex *col, double *d1, double *d2, mwSize n){
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int warpId = threadId / 32;
  int lane = threadId % 32;

  //the column (one warp per column)
  mwIndex j = warpId;

  double sum = 0;
  if(j < n){
    
    for(mwIndex id_i = col[j] + lane; id_i < col[j+1]; id_i += 32){

      // get the column (k)
      mwIndex k = row[id_i];
        
      // --- matrix-vector products
      sum += d1[k];
    }
  }

  sum = warp_reduce(sum);

  if(lane == 0 && j < n)
    d2[j] = sum - d1[j];
}

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

__global__ void compute_c3_Kernel(mwIndex *row, mwIndex *col, mwSize n, mwSize m, double* d4){
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int j, jp, k, l, x;

  //for each column (i)
  if(i < n){
    //for each row (j) that column (i) has an element
    for(jp = col[i]; jp < col[i+1]; jp++){
      j = row[jp];
      x = col[i];

      for(k = col[j]; k < col[j + 1]; k++){
        for(l = x; l < col[i+1]; l++){

          if(row[k] < row[l]){
            x = l;
            break;
          }
          else if(row[k] == row[l]){
            d4[i]++;
            x = l + 1;
            break;
          }
          else{
            x = l + 1;
          }
        }
      }
    }

    d4[i] /= 2;
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

  //------------if d2 vector kernel
  dim3 dimBlock2(BLOCK_SIZE); //thread per block
  dim3 dimGrid2(n); //num of blocks (and warps)
  //----------------------
  compute_d2_Kernel<<<dimGrid, dimBlock>>>(rowD, colD, d1D, d2D, n);

  cudaMemcpy(d2, d2D, n * sizeof(double), cudaMemcpyDeviceToHost);

  //d4
  cudaMalloc((void**)&d4D, n * sizeof(double));

  compute_c3_Kernel<<<dimGrid, dimBlock>>>(rowD, colD, n, m, d4D);

  cudaMemcpy(d4, d4D, n * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d1D);
  cudaFree(d2D);
  cudaFree(d4D);
  cudaFree(colD);
  cudaFree(rowD);

  //--------------------------------------------

  FILE* f;
  f = fopen("raw_results_cuda.txt", "w");

  for(mwSize i = 0; i < n; i++){
    d[2][i] = d[2][i] -  2 * d[4][i];
    d[3][i] -= d[4][i];
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